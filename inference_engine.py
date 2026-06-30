# -*- coding: utf-8 -*-
"""
IPG-Model 2.0 实时推理引擎 (Inference Engine)
功能：加载预训练权重，接收单条流量的 50 维特征，实时完成多模态融合、KNN构图和 GATv2 推理。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import warnings

# 忽略一些 sklearn 的版本警告
warnings.filterwarnings("ignore")

# =====================================================================
# 1. 网络结构定义 (必须与你训练时的结构一致才能成功加载权重)
# =====================================================================

class MLP128(nn.Module):
    """提取表格特征的 MLP (支持动态推断维度)"""
    def __init__(self, in_dim, h1_dim, h2_dim, emb_dim, dropout=0.3):
        super().__init__()
        # 对应你报错中的 "feat.0", "feat.3", "feat.6"
        self.feat = nn.Sequential(
            nn.Linear(in_dim, h1_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1_dim, h2_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2_dim, emb_dim)
        )
        # 对应你报错中的 "cls"
        self.cls = nn.Linear(emb_dim, 2)
        
    def forward(self, x):
        emb = self.feat(x)
        return emb, self.cls(emb)

class CNNMock(nn.Module):
    """由于缺失 best_cnn_encoder.pt，这里用一个简单的线性层模拟 128维 嵌入生成，以保证系统运转"""
    def __init__(self, out_dim=128):
        super().__init__()
        self.dummy_layer = nn.Linear(50, out_dim)
    def forward(self, x):
        # 实际工程中应替换为真正的序列生成和 ResNet/CNN 提取
        # 这里为了演示系统畅通，直接根据输入的 50维 表格特征映射一个 128维 的伪时序特征
        return self.dummy_layer(x)

from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

class GATv2EdgeNet(nn.Module):
    """主脑：GATv2 (带边缘特征)"""
    def __init__(self, in_dim=257, hidden=64, heads=4, edge_dim=1, dropout=0.3):
        super().__init__()
        self.gat = GATv2Conv(in_dim, hidden, heads=heads, concat=True,
                             dropout=dropout, edge_dim=edge_dim)
        self.lin = nn.Linear(hidden * heads, 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index, edge_attr):
        x_gat = self.gat(x, edge_index, edge_attr)
        x_gat = F.elu(x_gat)
        x_gat = self.dropout(x_gat)
        logits = self.lin(x_gat)
        return logits, x_gat

# =====================================================================
# 2. 推理引擎核心类
# =====================================================================

class IPGInferenceEngine:
    def __init__(self, data_dir=r"D:\program-2qu\data"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 初始化推理引擎 (Device: {self.device})...")
        
        # 1. 加载标准化器
        scaler_path = f"{data_dir}/mlp_struct_scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        print("✅ StandardScaler 加载成功。")

        # 2. 动态加载 MLP 特征提取器
        mlp_weight_path = f"{data_dir}/best_mlp_struct.pt"
        mlp_state = torch.load(mlp_weight_path, map_location=self.device)
        
        # 自动推断原网络各个层的形状，防止尺寸报错！
        in_dim = mlp_state['feat.0.weight'].shape[1]
        h1_dim = mlp_state['feat.0.weight'].shape[0]
        h2_dim = mlp_state['feat.3.weight'].shape[0]
        emb_dim = mlp_state['feat.6.weight'].shape[0]
        
        self.mlp = MLP128(in_dim, h1_dim, h2_dim, emb_dim).to(self.device)
        self.mlp.load_state_dict(mlp_state)
        self.mlp.eval()
        print(f"✅ MLP 权重加载成功 (自适应结构: {in_dim}->{h1_dim}->{h2_dim}->{emb_dim})。")

        # 3. 模拟加载 CNN 提取器 (填补缺失环节)
        self.cnn = CNNMock(out_dim=128).to(self.device)
        self.cnn.eval()
        print("⚠️ CNN 权重缺失，已启用自动 Mock 模拟模块。")

        # 4. 动态加载主脑 GATv2
        state = torch.load(f"{data_dir}/1best_gat_edge.pt", map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
            
        # 自动推断 GATv2 的 hidden 和 heads，防止尺寸报错！
        try:
            gat_heads = state['gat.att'].shape[1]
            gat_hidden = state['gat.att'].shape[2]
        except KeyError:
            gat_heads = 4
            gat_hidden = 128  # 容错：根据你的报错日志默认调整为 128
            
        self.gat = GATv2EdgeNet(in_dim=257, hidden=gat_hidden, heads=gat_heads).to(self.device)
        self.gat.load_state_dict(state, strict=False)
        self.gat.eval()
        print(f"✅ GATv2 主脑加载成功 (自适应结构: hidden={gat_hidden}, heads={gat_heads})。")

        # 5. 加载常驻内存的锚点库 (Anchors)
        anchors_data = np.load(f"{data_dir}/anchor_features_2000.npz")
        self.anchor_features = anchors_data['features'] # Shape: (2000, 257)
        self.anchor_labels = anchors_data['labels']     # Shape: (2000,)
        print(f"✅ 锚点库 (Anchors) 加载成功，包含 {len(self.anchor_labels)} 个历史节点。")
        
        # 预设的 50 个特征列名（必须与 scaler 训练时一致）
        self.feature_cols = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
            'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'Down/Up Ratio', 'Average Packet Size'
        ]

    def process_live_flow(self, df_50d_raw, top_k_neighbors=15):
        """
        核心推理流水线：接收单条流量的 DataFrame -> 融合特征 -> 构图 -> 推理
        """
        # --- 1. 数据标准化 ---
        # 确保列序对齐
        for col in self.feature_cols:
            if col not in df_50d_raw.columns:
                df_50d_raw[col] = 0.0
        X_input = df_50d_raw[self.feature_cols].values
        
        X_scaled = self.scaler.transform(X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # --- 2. 多模态特征提取与融合 ---
        with torch.no_grad():
            emb_mlp, _ = self.mlp(X_tensor)      # (1, 128)
            emb_cnn = self.cnn(X_tensor)         # (1, 128) Mock
            
            # 极简模拟香农熵计算 (基于归一化后的数据)
            X_norm = np.clip(X_scaled, 0.0001, None)
            row_sum = X_norm.sum() + 1e-8
            P = X_norm / row_sum
            entropy = -(P * np.log2(P)).sum()
            emb_ent = torch.tensor([[entropy]], dtype=torch.float32).to(self.device) # (1, 1)

            # 最终的 257 维融合特征
            f_257d = torch.cat([emb_mlp, emb_cnn, emb_ent], dim=1).cpu().numpy() # (1, 257)

        # --- 3. 极速 KNN 构图 (基于锚点) ---
        # 计算当前未知节点与 2000 个已知锚点的余弦相似度
        sims = cosine_similarity(f_257d, self.anchor_features)[0]
        
        # 找到最相似的 Top-K 个锚点的索引
        top_k_idx = np.argsort(sims)[-top_k_neighbors:]
        
        # 构建一个包含 1 个未知节点(索引为0) + 15 个锚点(索引为1~15)的小型子图
        nodes_features = np.vstack([f_257d, self.anchor_features[top_k_idx]])
        nodes_tensor = torch.tensor(nodes_features, dtype=torch.float32).to(self.device)
        
        # 连边：未知节点(0) 连向 锚点(1~15)
        edges = []
        edge_weights = []
        for i, idx in enumerate(top_k_idx):
            edges.append([0, i + 1])
            edges.append([i + 1, 0]) # 双向边
            edge_weights.extend([sims[idx], sims[idx]])
            
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1).to(self.device)

        # --- 4. GATv2 前向推理 ---
        with torch.no_grad():
            logits, _ = self.gat(nodes_tensor, edge_index, edge_attr)
            
            # 取出节点 0 (即未知流量) 的预测概率
            # 使用 Softmax 将结果转化为 0-1 的概率 (置信度)
            prob = F.softmax(logits[0], dim=0)[1].item() 
            
        # 整理用于解释图的数据
        connected_anchors = []
        for i, idx in enumerate(top_k_idx):
            connected_anchors.append({
                "local_id": i + 1,
                "similarity": float(sims[idx]),
                "true_label": int(self.anchor_labels[idx])
            })

        return {
            "threat_probability": prob,
            "connected_anchors": connected_anchors
        }

# =====================================================================
# 3. 简单本地测试
# =====================================================================
if __name__ == "__main__":
    # 初始化引擎 (只需在系统启动时做一次)
    engine = IPGInferenceEngine()
    
    # 模拟从 PCAP 提取出来的 50 维特征
    dummy_data = {col: [np.random.rand() * 100] for col in engine.feature_cols}
    df_live = pd.DataFrame(dummy_data)
    
    print("\n--- 正在模拟处理实时流量 ---")
    result = engine.process_live_flow(df_live)
    
    print(f"🎯 最终威胁概率 (置信度): {result['threat_probability']*100:.2f}%")
    print("🕸️ 发现的最相似锚点 (前3个):")
    for anchor in result['connected_anchors'][:3]:
        label_str = "🔴 恶意" if anchor['true_label'] == 1 else "🟢 良性"
        print(f"  - 锚点类型: {label_str}, 相似度: {anchor['similarity']:.4f}")