# -*- coding: utf-8 -*-
"""
研电赛工程化重构：PCAP 实时特征提取模块
目标：替代外部的 CICFlowMeter，直接在 Python 内存中从单条 PCAP 提取模型所需的 50 维统计特征。
依赖：pip install scapy pandas numpy
"""

import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
import time

# 这里列出你在 redeal.py 互信息筛选后，最终保留的那 50 个关键特征的名称
# (注意：在实际集成时，请替换为你真实选出的 50 个表头，这里提供典型示例)
TARGET_50_FEATURES = [
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

def extract_features_from_pcap(pcap_path):
    """
    读取单条 PCAP 流量文件，并在内存中计算统计特征，返回适配模型的 50 维 DataFrame。
    """
    print(f"📡 正在解析 PCAP 文件: {pcap_path}")
    start_time = time.time()
    
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        raise ValueError(f"无法读取 PCAP 文件: {e}")

    if not packets:
        raise ValueError("PCAP 文件为空")

    # --- 1. 初始化统计变量 ---
    fwd_pkts, bwd_pkts = [], []
    fwd_iats, bwd_iats, flow_iats = [], [], []
    
    # 简单的方向判断：以第一个包的源IP为 Fwd (前向) 方向
    first_pkt = packets[0]
    if IP in first_pkt:
        src_ip = first_pkt[IP].src
    else:
        src_ip = None

    last_time = float(first_pkt.time)
    fwd_last_time, bwd_last_time = last_time, last_time

    # 标志位统计
    flags_count = {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0}

    # --- 2. 遍历数据包进行基础统计 ---
    for pkt in packets:
        current_time = float(pkt.time)
        pkt_len = len(pkt)
        
        # 计算总体 IAT (Inter-Arrival Time)
        if current_time > last_time:
            flow_iats.append(current_time - last_time)
        last_time = current_time

        # 提取 TCP 标志位
        if TCP in pkt:
            flags = pkt[TCP].flags
            if flags & 0x01: flags_count['FIN'] += 1
            if flags & 0x02: flags_count['SYN'] += 1
            if flags & 0x04: flags_count['RST'] += 1
            if flags & 0x08: flags_count['PSH'] += 1
            if flags & 0x10: flags_count['ACK'] += 1
            if flags & 0x20: flags_count['URG'] += 1

        # 方向判断与统计
        if IP in pkt and pkt[IP].src == src_ip:
            # 前向包 (Fwd)
            fwd_pkts.append(pkt_len)
            if current_time > fwd_last_time:
                fwd_iats.append(current_time - fwd_last_time)
            fwd_last_time = current_time
        else:
            # 后向包 (Bwd)
            bwd_pkts.append(pkt_len)
            if current_time > bwd_last_time:
                bwd_iats.append(current_time - bwd_last_time)
            bwd_last_time = current_time

    # --- 3. 计算聚合特征 (类似于 CICFlowMeter) ---
    flow_duration = float(packets[-1].time - packets[0].time) * 1e6 # 转换为微秒
    all_pkts = fwd_pkts + bwd_pkts
    
    def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
    def safe_std(lst): return float(np.std(lst)) if len(lst) > 1 else 0.0
    def safe_max(lst): return float(np.max(lst)) if lst else 0.0
    def safe_min(lst): return float(np.min(lst)) if lst else 0.0
    
    # 构建特征字典
    raw_features = {
        'Flow Duration': flow_duration,
        'Total Fwd Packets': len(fwd_pkts),
        'Total Backward Packets': len(bwd_pkts),
        'Total Length of Fwd Packets': sum(fwd_pkts),
        'Total Length of Bwd Packets': sum(bwd_pkts),
        
        'Fwd Packet Length Max': safe_max(fwd_pkts),
        'Fwd Packet Length Min': safe_min(fwd_pkts),
        'Fwd Packet Length Mean': safe_mean(fwd_pkts),
        'Fwd Packet Length Std': safe_std(fwd_pkts),
        
        'Bwd Packet Length Max': safe_max(bwd_pkts),
        'Bwd Packet Length Min': safe_min(bwd_pkts),
        'Bwd Packet Length Mean': safe_mean(bwd_pkts),
        'Bwd Packet Length Std': safe_std(bwd_pkts),
        
        'Flow Bytes/s': sum(all_pkts) / (flow_duration / 1e6) if flow_duration > 0 else 0.0,
        'Flow Packets/s': len(all_pkts) / (flow_duration / 1e6) if flow_duration > 0 else 0.0,
        
        'Flow IAT Mean': safe_mean(flow_iats) * 1e6,
        'Flow IAT Std': safe_std(flow_iats) * 1e6,
        'Flow IAT Max': safe_max(flow_iats) * 1e6,
        'Flow IAT Min': safe_min(flow_iats) * 1e6,
        
        'Fwd IAT Total': sum(fwd_iats) * 1e6,
        'Fwd IAT Mean': safe_mean(fwd_iats) * 1e6,
        'Fwd IAT Std': safe_std(fwd_iats) * 1e6,
        'Fwd IAT Max': safe_max(fwd_iats) * 1e6,
        'Fwd IAT Min': safe_min(fwd_iats) * 1e6,

        'Bwd IAT Total': sum(bwd_iats) * 1e6,
        'Bwd IAT Mean': safe_mean(bwd_iats) * 1e6,
        'Bwd IAT Std': safe_std(bwd_iats) * 1e6,
        'Bwd IAT Max': safe_max(bwd_iats) * 1e6,
        'Bwd IAT Min': safe_min(bwd_iats) * 1e6,
        
        # 简化标志位和其他网络头特征 (此处设为示例或0，实际可根据报文深入提取)
        'Fwd PSH Flags': 0, 'Bwd PSH Flags': 0, 
        'Fwd URG Flags': 0, 'Bwd URG Flags': 0,
        'Fwd Header Length': len(fwd_pkts) * 20, # 估算
        'Bwd Header Length': len(bwd_pkts) * 20,
        
        'Fwd Packets/s': len(fwd_pkts) / (flow_duration / 1e6) if flow_duration > 0 else 0.0,
        'Bwd Packets/s': len(bwd_pkts) / (flow_duration / 1e6) if flow_duration > 0 else 0.0,
        
        'Min Packet Length': safe_min(all_pkts),
        'Max Packet Length': safe_max(all_pkts),
        'Packet Length Mean': safe_mean(all_pkts),
        'Packet Length Std': safe_std(all_pkts),
        'Packet Length Variance': safe_std(all_pkts) ** 2,
        
        'FIN Flag Count': flags_count['FIN'],
        'SYN Flag Count': flags_count['SYN'],
        'RST Flag Count': flags_count['RST'],
        'PSH Flag Count': flags_count['PSH'],
        'ACK Flag Count': flags_count['ACK'],
        'URG Flag Count': flags_count['URG'],
        
        'Down/Up Ratio': len(bwd_pkts) / len(fwd_pkts) if len(fwd_pkts) > 0 else 0.0,
        'Average Packet Size': safe_mean(all_pkts)
    }

    # --- 4. 截取并对齐模型需要的 50 维特征 ---
    final_features = {}
    for col in TARGET_50_FEATURES:
        # 如果提出来的特征字典里有，就用；没有就补 0 (容错机制)
        final_features[col] = [raw_features.get(col, 0.0)]
        
    df_50d = pd.DataFrame(final_features)
    
    print(f"✅ 提取完成! 耗时: {time.time() - start_time:.3f}s, 输出维度: {df_50d.shape}")
    return df_50d

# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟测试：你需要在这里放一个真实的短 pcap 文件路径
    test_pcap = "sample_traffic.pcap" 
    try:
        # 假设当前目录下有这个文件才执行
        df_result = extract_features_from_pcap(test_pcap)
        print("提取出的前 5 个特征预览:")
        print(df_result.iloc[:, :5].to_dict('records')[0])
    except Exception as e:
        print(f"测试跳过 (原因: {e})，请准备真实的 pcap 文件进行集成测试。")