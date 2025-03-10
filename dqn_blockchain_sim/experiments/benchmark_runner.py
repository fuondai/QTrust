"""
Script thực hiện các thử nghiệm mở rộng để đánh giá hiệu suất của các thuật toán
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import numpy as np
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_blockchain_sim.simulation.advanced_simulation import AdvancedSimulation
from dqn_blockchain_sim.simulation.basic_simulation import BasicSimulation


def run_benchmark(config: Dict[str, Any], output_dir: str = "benchmark_results") -> Dict[str, Any]:
    """
    Chạy một thử nghiệm với cấu hình cho trước
    
    Args:
        config: Cấu hình thử nghiệm
        output_dir: Thư mục lưu kết quả
        
    Returns:
        Kết quả thử nghiệm
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo ID cho thử nghiệm
    benchmark_id = f"{config['name']}_{int(time.time())}"
    
    # Lưu cấu hình
    config_file = os.path.join(output_dir, f"{benchmark_id}_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Khởi tạo mô phỏng
    if config.get('use_advanced', True):
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=config.get('use_real_data', False),
            use_dqn=config.get('use_dqn', True),
            eth_api_key=config.get('eth_api_key', None),
            data_dir=config.get('data_dir', "data"),
            log_dir=output_dir
        )
    else:
        simulation = BasicSimulation(
            num_shards=config.get('num_shards', 8),
            use_dqn=config.get('use_dqn', False)
        )
    
    # Chạy mô phỏng
    print(f"Running benchmark: {config['name']}")
    results = simulation.run_simulation(
        num_steps=config.get('num_steps', 100),
        tx_per_step=config.get('tx_per_step', 20),
        visualize=False,
        save_stats=True
    )
    
    # Lưu kết quả
    results_file = os.path.join(output_dir, f"{benchmark_id}_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Trả về kết quả
    return {
        'config': config,
        'results': results,
        'benchmark_id': benchmark_id
    }


def run_comparative_benchmarks() -> List[Dict[str, Any]]:
    """
    Chạy các thử nghiệm so sánh giữa các phương pháp khác nhau
    
    Returns:
        Danh sách kết quả thử nghiệm
    """
    benchmarks = []
    
    # Cấu hình cơ bản
    base_config = {
        'num_steps': 100,
        'tx_per_step': 20,
        'data_dir': "data",
        'use_real_data': False
    }
    
    # 1. So sánh số lượng shard
    shard_configs = []
    for num_shards in [4, 8, 16, 32]:
        config = base_config.copy()
        config.update({
            'name': f"shard_comparison_{num_shards}",
            'num_shards': num_shards,
            'use_dqn': True,
            'use_advanced': True
        })
        shard_configs.append(config)
    
    # 2. So sánh với và không có DQN
    dqn_configs = []
    for use_dqn in [True, False]:
        config = base_config.copy()
        config.update({
            'name': f"dqn_comparison_{'with' if use_dqn else 'without'}",
            'num_shards': 8,
            'use_dqn': use_dqn,
            'use_advanced': True
        })
        dqn_configs.append(config)
    
    # 3. So sánh các thuật toán đồng thuận
    consensus_configs = []
    for consensus_type in ['standard', 'adaptive']:
        config = base_config.copy()
        config.update({
            'name': f"consensus_comparison_{consensus_type}",
            'num_shards': 8,
            'use_dqn': True,
            'use_advanced': True,
            'consensus_type': consensus_type
        })
        consensus_configs.append(config)
    
    # 4. So sánh khối lượng giao dịch
    load_configs = []
    for tx_per_step in [10, 20, 50, 100]:
        config = base_config.copy()
        config.update({
            'name': f"load_comparison_{tx_per_step}",
            'num_shards': 8,
            'tx_per_step': tx_per_step,
            'use_dqn': True,
            'use_advanced': True
        })
        load_configs.append(config)
    
    # Kết hợp tất cả cấu hình
    all_configs = shard_configs + dqn_configs + consensus_configs + load_configs
    
    # Chạy tất cả thử nghiệm
    for config in tqdm(all_configs, desc="Running benchmarks"):
        result = run_benchmark(config)
        benchmarks.append(result)
    
    return benchmarks


def analyze_results(benchmarks: List[Dict[str, Any]], output_dir: str = "benchmark_results") -> None:
    """
    Phân tích kết quả thử nghiệm và tạo biểu đồ
    
    Args:
        benchmarks: Danh sách kết quả thử nghiệm
        output_dir: Thư mục lưu kết quả
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo DataFrame từ kết quả
    results_data = []
    for benchmark in benchmarks:
        config = benchmark['config']
        results = benchmark['results']
        
        # Trích xuất các metrics chính
        row = {
            'name': config['name'],
            'num_shards': config.get('num_shards', 8),
            'tx_per_step': config.get('tx_per_step', 20),
            'use_dqn': config.get('use_dqn', True),
            'use_advanced': config.get('use_advanced', True),
            'consensus_type': config.get('consensus_type', 'adaptive'),
            'throughput': results.get('avg_throughput', 0),
            'latency': results.get('avg_latency', 0),
            'success_rate': results.get('success_rate', 0),
            'congestion': results.get('avg_congestion', 0),
            'energy_consumption': results.get('energy_consumption', 0)
        }
        results_data.append(row)
    
    # Tạo DataFrame
    df = pd.DataFrame(results_data)
    
    # Lưu DataFrame
    df.to_csv(os.path.join(output_dir, "benchmark_summary.csv"), index=False)
    
    # Tạo các biểu đồ so sánh
    create_comparison_charts(df, output_dir)


def create_comparison_charts(df: pd.DataFrame, output_dir: str) -> None:
    """
    Tạo các biểu đồ so sánh từ DataFrame kết quả
    
    Args:
        df: DataFrame kết quả
        output_dir: Thư mục lưu biểu đồ
    """
    # 1. So sánh số lượng shard
    shard_df = df[df['name'].str.contains('shard_comparison')]
    if not shard_df.empty:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.bar(shard_df['num_shards'].astype(str), shard_df['throughput'])
        plt.title('Throughput vs Number of Shards')
        plt.xlabel('Number of Shards')
        plt.ylabel('Throughput (tx/step)')
        
        plt.subplot(2, 2, 2)
        plt.bar(shard_df['num_shards'].astype(str), shard_df['latency'])
        plt.title('Latency vs Number of Shards')
        plt.xlabel('Number of Shards')
        plt.ylabel('Latency (ms)')
        
        plt.subplot(2, 2, 3)
        plt.bar(shard_df['num_shards'].astype(str), shard_df['success_rate'])
        plt.title('Success Rate vs Number of Shards')
        plt.xlabel('Number of Shards')
        plt.ylabel('Success Rate')
        
        plt.subplot(2, 2, 4)
        plt.bar(shard_df['num_shards'].astype(str), shard_df['energy_consumption'])
        plt.title('Energy Consumption vs Number of Shards')
        plt.xlabel('Number of Shards')
        plt.ylabel('Energy Consumption')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shard_comparison.png"))
    
    # 2. So sánh với và không có DQN
    dqn_df = df[df['name'].str.contains('dqn_comparison')]
    if not dqn_df.empty:
        plt.figure(figsize=(12, 8))
        
        metrics = ['throughput', 'latency', 'success_rate', 'energy_consumption']
        titles = ['Throughput', 'Latency', 'Success Rate', 'Energy Consumption']
        ylabels = ['Throughput (tx/step)', 'Latency (ms)', 'Success Rate', 'Energy Consumption']
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            plt.subplot(2, 2, i+1)
            plt.bar(dqn_df['use_dqn'].map({True: 'With DQN', False: 'Without DQN'}), dqn_df[metric])
            plt.title(f'{title} with/without DQN')
            plt.ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dqn_comparison.png"))
    
    # 3. So sánh các thuật toán đồng thuận
    consensus_df = df[df['name'].str.contains('consensus_comparison')]
    if not consensus_df.empty:
        plt.figure(figsize=(12, 8))
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            plt.subplot(2, 2, i+1)
            plt.bar(consensus_df['consensus_type'], consensus_df[metric])
            plt.title(f'{title} by Consensus Type')
            plt.ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "consensus_comparison.png"))
    
    # 4. So sánh khối lượng giao dịch
    load_df = df[df['name'].str.contains('load_comparison')]
    if not load_df.empty:
        plt.figure(figsize=(12, 8))
        
        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
            plt.subplot(2, 2, i+1)
            plt.bar(load_df['tx_per_step'].astype(str), load_df[metric])
            plt.title(f'{title} vs Transaction Load')
            plt.xlabel('Transactions per Step')
            plt.ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "load_comparison.png"))


def main():
    """
    Hàm chính để chạy các thử nghiệm
    """
    # Tạo thư mục đầu ra
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Chạy các thử nghiệm so sánh
    benchmarks = run_comparative_benchmarks()
    
    # Phân tích kết quả
    analyze_results(benchmarks, output_dir)
    
    print(f"Benchmark results saved to {output_dir}")


if __name__ == "__main__":
    main() 