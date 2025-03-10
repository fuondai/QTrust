"""
Script chạy tất cả các phân tích và tạo báo cáo tổng hợp
"""

import os
import sys
import subprocess
import argparse
import time
from datetime import datetime

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dqn_blockchain_sim.experiments.benchmark_runner import run_comparative_benchmarks, analyze_results
from dqn_blockchain_sim.experiments.consensus_comparison import ConsensusSimulator
from dqn_blockchain_sim.experiments.performance_analysis import PerformanceAnalyzer
from dqn_blockchain_sim.experiments.generate_report import ReportGenerator


def setup_directories():
    """
    Tạo các thư mục cần thiết
    """
    os.makedirs("benchmark_results", exist_ok=True)
    os.makedirs("consensus_comparison", exist_ok=True)
    os.makedirs("performance_analysis", exist_ok=True)
    os.makedirs("final_report", exist_ok=True)


def run_benchmarks(args):
    """
    Chạy các benchmark
    
    Args:
        args: Tham số dòng lệnh
    """
    print("\n" + "="*50)
    print("Chạy benchmarks...")
    print("="*50)
    
    # Tham số benchmark
    num_configs = args.num_configs if hasattr(args, 'num_configs') else 3
    
    # Chạy benchmark
    benchmarks = run_comparative_benchmarks(num_configs=num_configs)
    analyze_results(benchmarks, output_dir="benchmark_results")
    
    print("Benchmarks hoàn thành.")


def run_consensus_comparison(args):
    """
    Chạy so sánh các phương pháp đồng thuận
    
    Args:
        args: Tham số dòng lệnh
    """
    print("\n" + "="*50)
    print("Chạy so sánh các phương pháp đồng thuận...")
    print("="*50)
    
    # Khởi tạo simulator
    simulator = ConsensusSimulator(output_dir="consensus_comparison")
    
    # Tham số cho phương pháp đồng thuận
    config = {
        'num_shards': 4,
        'num_nodes': 20,
        'num_validators': 10,
        'num_transactions': 100,
        'block_size': 10
    }
    
    # Chạy so sánh
    protocols = ["PoW", "PoS", "PBFT", "ACSC"]
    simulator.run_comparison(protocols=protocols, config=config)
    
    # Tạo biểu đồ
    simulator.visualize_comparison()
    
    print("So sánh các phương pháp đồng thuận hoàn thành.")


def run_performance_analysis(args):
    """
    Chạy phân tích hiệu suất
    
    Args:
        args: Tham số dòng lệnh
    """
    print("\n" + "="*50)
    print("Chạy phân tích hiệu suất...")
    print("="*50)
    
    # Khởi tạo analyzer
    analyzer = PerformanceAnalyzer(output_dir="performance_analysis")
    
    # Tạo các cấu hình mô phỏng
    configs = []
    
    # 1. Cấu hình với số lượng shard khác nhau
    for num_shards in [4, 8]:
        config = {
            'name': f"shard_analysis_{num_shards}",
            'num_shards': num_shards,
            'num_steps': 50,
            'tx_per_step': 10,
            'use_dqn': True
        }
        configs.append(config)
    
    # 2. Cấu hình với khối lượng giao dịch khác nhau
    for tx_per_step in [10, 20]:
        config = {
            'name': f"load_analysis_{tx_per_step}",
            'num_shards': 4,
            'num_steps': 50,
            'tx_per_step': tx_per_step,
            'use_dqn': True
        }
        configs.append(config)
    
    # 3. Cấu hình với và không có DQN
    for use_dqn in [True, False]:
        config = {
            'name': f"dqn_analysis_{'with' if use_dqn else 'without'}",
            'num_shards': 4,
            'num_steps': 50,
            'tx_per_step': 10,
            'use_dqn': use_dqn
        }
        configs.append(config)
    
    # Chạy phân tích
    analyzer.run_analysis(configs)
    
    print("Phân tích hiệu suất hoàn thành.")


def generate_final_report():
    """
    Tạo báo cáo tổng hợp
    """
    print("\n" + "="*50)
    print("Tạo báo cáo tổng hợp...")
    print("="*50)
    
    # Khởi tạo generator
    generator = ReportGenerator(output_dir="final_report")
    
    # Tạo báo cáo
    report_file = generator.generate_report()
    
    print(f"Báo cáo đã được tạo: {report_file}")


def main():
    """
    Hàm chính để chạy tất cả các phân tích
    """
    # Tạo parser dòng lệnh
    parser = argparse.ArgumentParser(description="Chạy tất cả các phân tích và tạo báo cáo tổng hợp")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Bỏ qua benchmark")
    parser.add_argument("--skip-consensus", action="store_true", help="Bỏ qua so sánh đồng thuận")
    parser.add_argument("--skip-performance", action="store_true", help="Bỏ qua phân tích hiệu suất")
    parser.add_argument("--num-configs", type=int, default=3, help="Số lượng cấu hình cho benchmark")
    args = parser.parse_args()
    
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # In thông tin bắt đầu
    print(f"Bắt đầu phân tích: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Tạo các thư mục cần thiết
    setup_directories()
    
    # Chạy các phân tích
    if not args.skip_benchmarks:
        run_benchmarks(args)
    else:
        print("Đã bỏ qua benchmarks.")
        
    if not args.skip_consensus:
        run_consensus_comparison(args)
    else:
        print("Đã bỏ qua so sánh đồng thuận.")
        
    if not args.skip_performance:
        run_performance_analysis(args)
    else:
        print("Đã bỏ qua phân tích hiệu suất.")
    
    # Tạo báo cáo tổng hợp
    generate_final_report()
    
    # Tính thời gian chạy
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"Tất cả các phân tích đã hoàn thành trong: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("="*50)


if __name__ == "__main__":
    main() 