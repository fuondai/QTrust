import os
import sys

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')  # Sử dụng backend không tương tác
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

from experiments.performance_analysis import PerformanceAnalyzer

def main():
    try:
        print("Bắt đầu phân tích blockchain...")
        
        # Tạo thư mục kết quả
        os.makedirs("benchmark_results", exist_ok=True)
        os.makedirs("consensus_comparison", exist_ok=True) 
        os.makedirs("performance_analysis", exist_ok=True)
        os.makedirs("final_report", exist_ok=True)

        # Khởi tạo analyzer
        analyzer = PerformanceAnalyzer(output_dir="performance_analysis")
        
        # Cấu hình mô phỏng cơ bản
        basic_config = {
            "num_shards": 4,
            "nodes_per_shard": 10,
            "num_transactions": 1000,
            "simulation_steps": 100,
            "consensus_protocol": "PBFT"
        }
        
        # Cấu hình so sánh các phương pháp đồng thuận
        consensus_configs = [
            {"consensus_protocol": "PoW", **basic_config},
            {"consensus_protocol": "PoS", **basic_config},
            {"consensus_protocol": "PBFT", **basic_config},
            {"consensus_protocol": "ACSC", **basic_config}
        ]
        
        # Chạy phân tích
        print("Đang chạy phân tích hiệu năng...")
        analyzer.run_analysis(consensus_configs)
        
        print("Phân tích hoàn tất. Kết quả được lưu trong thư mục 'performance_analysis'")
        
    except Exception as e:
        print(f"Lỗi trong quá trình phân tích: {str(e)}")
        raise

if __name__ == "__main__":
    main() 