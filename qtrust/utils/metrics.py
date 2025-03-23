"""
Module chứa các hàm đánh giá hiệu suất cho hệ thống QTrust.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

def calculate_transaction_throughput(successful_txs: int, total_time: float) -> float:
    """
    Tính toán thông lượng giao dịch.
    
    Args:
        successful_txs: Số lượng giao dịch thành công
        total_time: Tổng thời gian (ms)
        
    Returns:
        float: Thông lượng giao dịch (giao dịch/ms)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_latency_metrics(latencies: List[float]) -> Dict[str, float]:
    """
    Tính toán các chỉ số về độ trễ.
    
    Args:
        latencies: Danh sách độ trễ của các giao dịch (ms)
        
    Returns:
        Dict: Các thông số về độ trễ (trung bình, trung vị, cao nhất, v.v.)
    """
    if not latencies:
        return {
            'avg_latency': 0.0,
            'median_latency': 0.0,
            'min_latency': 0.0,
            'max_latency': 0.0,
            'p95_latency': 0.0,
            'p99_latency': 0.0
        }
    
    return {
        'avg_latency': np.mean(latencies),
        'median_latency': np.median(latencies),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies),
        'p95_latency': np.percentile(latencies, 95),
        'p99_latency': np.percentile(latencies, 99)
    }

def calculate_energy_efficiency(energy_consumption: float, successful_txs: int) -> float:
    """
    Tính toán hiệu suất năng lượng.
    
    Args:
        energy_consumption: Tổng năng lượng tiêu thụ
        successful_txs: Số lượng giao dịch thành công
        
    Returns:
        float: Năng lượng tiêu thụ trên mỗi giao dịch thành công
    """
    if successful_txs == 0:
        return float('inf')
    return energy_consumption / successful_txs

def calculate_security_metrics(
    trust_scores: Dict[int, float], 
    malicious_nodes: List[int]
) -> Dict[str, float]:
    """
    Tính toán các chỉ số về bảo mật.
    
    Args:
        trust_scores: Điểm tin cậy của các node
        malicious_nodes: Danh sách ID của các node độc hại
        
    Returns:
        Dict: Các thông số về bảo mật
    """
    total_nodes = len(trust_scores)
    if total_nodes == 0:
        return {
            'avg_trust': 0.0,
            'malicious_ratio': 0.0,
            'trust_variance': 0.0
        }
    
    avg_trust = np.mean(list(trust_scores.values()))
    trust_variance = np.var(list(trust_scores.values()))
    malicious_ratio = len(malicious_nodes) / total_nodes if total_nodes > 0 else 0
    
    return {
        'avg_trust': avg_trust,
        'malicious_ratio': malicious_ratio,
        'trust_variance': trust_variance
    }

def calculate_cross_shard_metrics(
    cross_shard_txs: int, 
    total_txs: int, 
    cross_shard_latencies: List[float],
    intra_shard_latencies: List[float]
) -> Dict[str, float]:
    """
    Tính toán các chỉ số về giao dịch xuyên shard.
    
    Args:
        cross_shard_txs: Số lượng giao dịch xuyên shard
        total_txs: Tổng số giao dịch
        cross_shard_latencies: Độ trễ của các giao dịch xuyên shard
        intra_shard_latencies: Độ trễ của các giao dịch trong shard
        
    Returns:
        Dict: Các thông số về giao dịch xuyên shard
    """
    cross_shard_ratio = cross_shard_txs / total_txs if total_txs > 0 else 0
    
    cross_shard_avg_latency = np.mean(cross_shard_latencies) if cross_shard_latencies else 0
    intra_shard_avg_latency = np.mean(intra_shard_latencies) if intra_shard_latencies else 0
    
    latency_overhead = (cross_shard_avg_latency / intra_shard_avg_latency) if intra_shard_avg_latency > 0 else 0
    
    return {
        'cross_shard_ratio': cross_shard_ratio,
        'cross_shard_avg_latency': cross_shard_avg_latency,
        'intra_shard_avg_latency': intra_shard_avg_latency,
        'latency_overhead': latency_overhead
    }

def generate_performance_report(metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Tạo báo cáo hiệu suất từ các chỉ số.
    
    Args:
        metrics: Dictionary chứa các chỉ số hiệu suất
        
    Returns:
        pd.DataFrame: Báo cáo hiệu suất dạng bảng
    """
    report = pd.DataFrame({
        'Metric': [
            'Throughput (tx/s)',
            'Average Latency (ms)',
            'Median Latency (ms)',
            'P95 Latency (ms)',
            'Energy per Transaction',
            'Average Trust Score',
            'Malicious Node Ratio',
            'Cross-Shard Transaction Ratio',
            'Cross-Shard Latency Overhead'
        ],
        'Value': [
            metrics.get('throughput', 0) * 1000,  # Chuyển đổi từ tx/ms sang tx/s
            metrics.get('latency', {}).get('avg_latency', 0),
            metrics.get('latency', {}).get('median_latency', 0),
            metrics.get('latency', {}).get('p95_latency', 0),
            metrics.get('energy_per_tx', 0),
            metrics.get('security', {}).get('avg_trust', 0),
            metrics.get('security', {}).get('malicious_ratio', 0),
            metrics.get('cross_shard', {}).get('cross_shard_ratio', 0),
            metrics.get('cross_shard', {}).get('latency_overhead', 0)
        ]
    })
    
    return report

def plot_trust_distribution(trust_scores: Dict[int, float], 
                           malicious_nodes: List[int], 
                           title: str = "Trust Score Distribution",
                           save_path: Optional[str] = None):
    """
    Vẽ đồ thị phân phối điểm tin cậy của các node.
    
    Args:
        trust_scores: Điểm tin cậy của các node
        malicious_nodes: Danh sách ID của các node độc hại
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(10, 6))
    
    # Tạo danh sách điểm tin cậy cho mỗi nhóm node
    normal_nodes = [node_id for node_id in trust_scores if node_id not in malicious_nodes]
    
    normal_scores = [trust_scores[node_id] for node_id in normal_nodes]
    malicious_scores = [trust_scores[node_id] for node_id in malicious_nodes if node_id in trust_scores]
    
    # Vẽ histogram
    sns.histplot(normal_scores, color='green', alpha=0.5, label='Normal Nodes', bins=20)
    if malicious_scores:
        sns.histplot(malicious_scores, color='red', alpha=0.5, label='Malicious Nodes', bins=20)
    
    plt.title(title)
    plt.xlabel('Trust Score')
    plt.ylabel('Number of Nodes')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_performance_comparison(results: Dict[str, Dict[str, List[float]]], 
                               metric_name: str,
                               title: str,
                               ylabel: str,
                               save_path: Optional[str] = None):
    """
    Vẽ đồ thị so sánh hiệu suất giữa các phương pháp.
    
    Args:
        results: Dictionary chứa kết quả của các phương pháp
        metric_name: Tên của chỉ số cần so sánh
        title: Tiêu đề của đồ thị
        ylabel: Nhãn trục y
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(12, 8))
    
    # Tạo dữ liệu cho boxplot
    data = []
    labels = []
    
    for method_name, method_results in results.items():
        if metric_name in method_results:
            data.append(method_results[metric_name])
            labels.append(method_name)
    
    # Vẽ boxplot
    box = plt.boxplot(data, patch_artist=True, labels=labels)
    
    # Đặt màu cho các hộp
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(box['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_time_series(data: List[float], 
                    title: str, 
                    xlabel: str, 
                    ylabel: str,
                    window: int = 10,
                    save_path: Optional[str] = None):
    """
    Vẽ đồ thị chuỗi thời gian.
    
    Args:
        data: Dữ liệu cần vẽ
        title: Tiêu đề của đồ thị
        xlabel: Nhãn trục x
        ylabel: Nhãn trục y
        window: Kích thước cửa sổ cho đường trung bình động
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(12, 6))
    
    # Vẽ dữ liệu gốc
    plt.plot(data, alpha=0.5, label='Raw Data')
    
    # Vẽ đường trung bình động
    if len(data) >= window:
        moving_avg = pd.Series(data).rolling(window=window).mean().values
        plt.plot(range(window-1, len(data)), moving_avg[window-1:], 'r-', linewidth=2, label=f'Moving Average (window={window})')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_heatmap(data: np.ndarray, 
                x_labels: List[str], 
                y_labels: List[str],
                title: str,
                save_path: Optional[str] = None):
    """
    Vẽ heatmap.
    
    Args:
        data: Dữ liệu dạng ma trận
        x_labels: Nhãn cho trục x
        y_labels: Nhãn cho trục y
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(data, annot=True, cmap='viridis', xticklabels=x_labels, yticklabels=y_labels)
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def calculate_throughput(successful_txs: int, total_time: float) -> float:
    """
    Tính toán thông lượng giao dịch.
    
    Args:
        successful_txs: Số lượng giao dịch thành công
        total_time: Tổng thời gian (s)
        
    Returns:
        float: Thông lượng giao dịch (giao dịch/giây)
    """
    if total_time == 0:
        return 0.0
    return successful_txs / total_time

def calculate_cross_shard_transaction_ratio(cross_shard_txs: int, total_txs: int) -> float:
    """
    Tính toán tỷ lệ giao dịch xuyên shard.
    
    Args:
        cross_shard_txs: Số lượng giao dịch xuyên shard
        total_txs: Tổng số giao dịch
        
    Returns:
        float: Tỷ lệ giao dịch xuyên shard
    """
    if total_txs == 0:
        return 0.0
    return cross_shard_txs / total_txs

def plot_performance_metrics(metrics: Dict[str, List[float]], 
                            title: str = "Performance Metrics Over Time",
                            save_path: Optional[str] = None):
    """
    Vẽ đồ thị các chỉ số hiệu suất theo thời gian.
    
    Args:
        metrics: Dictionary chứa dữ liệu của các chỉ số theo thời gian
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(15, 10))
    
    metric_names = list(metrics.keys())
    num_metrics = len(metric_names)
    
    rows = (num_metrics + 1) // 2  # Số hàng của lưới đồ thị
    cols = min(2, num_metrics)      # Số cột của lưới đồ thị
    
    for i, metric_name in enumerate(metric_names):
        plt.subplot(rows, cols, i+1)
        
        values = metrics[metric_name]
        x = range(len(values))
        
        plt.plot(x, values)
        plt.title(metric_name)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Thêm đường trung bình động
        window = min(10, len(values))
        if len(values) >= window:
            moving_avg = pd.Series(values).rolling(window=window).mean().values
            plt.plot(range(window-1, len(values)), moving_avg[window-1:], 'r-', linewidth=2, 
                    label=f'Moving Avg (w={window})')
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_comparison_charts(comparison_data: Dict[str, Dict[str, float]], 
                          metrics: List[str],
                          title: str = "Performance Comparison",
                          save_path: Optional[str] = None):
    """
    Vẽ đồ thị so sánh hiệu suất giữa các phương pháp.
    
    Args:
        comparison_data: Dictionary chứa dữ liệu so sánh giữa các phương pháp
        metrics: Danh sách các chỉ số cần so sánh
        title: Tiêu đề của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=(15, 10))
    
    num_metrics = len(metrics)
    rows = (num_metrics + 1) // 2
    cols = min(2, num_metrics)
    
    method_names = list(comparison_data.keys())
    
    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i+1)
        
        # Chuẩn bị dữ liệu
        values = [comparison_data[method][metric] for method in method_names]
        
        # Vẽ biểu đồ cột
        bars = plt.bar(method_names, values)
        
        # Thêm nhãn giá trị
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(metric)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_metrics_over_time(metrics_over_time: Dict[str, List[float]],
                           labels: List[str],
                           title: str = "Metrics Over Time",
                           xlabel: str = "Step",
                           figsize: Tuple[int, int] = (12, 8),
                           save_path: Optional[str] = None):
    """
    Vẽ đồ thị các chỉ số theo thời gian.
    
    Args:
        metrics_over_time: Dictionary chứa dữ liệu của các chỉ số theo thời gian
        labels: Nhãn cho từng chỉ số
        title: Tiêu đề của đồ thị
        xlabel: Nhãn trục x
        figsize: Kích thước của đồ thị
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    for i, (metric_name, values) in enumerate(metrics_over_time.items()):
        plt.subplot(len(metrics_over_time), 1, i+1)
        plt.plot(values)
        plt.ylabel(labels[i] if i < len(labels) else metric_name)
        if i == len(metrics_over_time) - 1:
            plt.xlabel(xlabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
    
    plt.suptitle(title, fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close() 