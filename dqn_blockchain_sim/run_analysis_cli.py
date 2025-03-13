import argparse
import os
import sys
import json
import glob
from datetime import datetime

# Thêm thư viện matplotlib nếu có thể import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Thư viện matplotlib không khả dụng. Các chức năng trực quan hóa sẽ bị vô hiệu.")

from simulation.advanced_simulation import AdvancedBlockchainSimulation
from experiments.performance_analysis import PerformanceAnalyzer
from utils.visualization import plot_performance_metrics

# Thêm thư viện cho việc phân tích mã nguồn
import inspect
import importlib
import pkgutil

def get_module_structure(package_name):
    """Lấy cấu trúc module từ tên gói"""
    package = importlib.import_module(package_name)
    module_structure = {}
    
    if hasattr(package, '__path__'):
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            if is_pkg:
                module_structure[name] = get_module_structure(name)
            else:
                try:
                    module = importlib.import_module(name)
                    module_structure[name] = {
                        'functions': [f for f, _ in inspect.getmembers(module, inspect.isfunction)],
                        'classes': [c for c, _ in inspect.getmembers(module, inspect.isclass)]
                    }
                except:
                    module_structure[name] = "Error loading module"
    
    return module_structure

def create_code_context():
    """Tạo báo cáo về cấu trúc và context của dự án"""
    print("Đang tạo báo cáo về cấu trúc code...")
    
    # Cấu trúc chính của dự án
    main_packages = ['agents', 'blockchain', 'simulation', 'utils', 'consensus', 'tdcm']
    code_structure = {}
    
    for package in main_packages:
        try:
            code_structure[package] = get_module_structure(package)
        except ImportError:
            code_structure[package] = "Package not found"
    
    # Tạo báo cáo về các class và mối quan hệ quan trọng
    relationships = {
        'blockchain_network': 'Core network class managing the blockchain',
        'shard': 'Represents a single shard in the network',
        'mad_rapid': 'Protocol for optimizing cross-shard transactions',
        'acsc': 'Adaptive Cross-Shard Consensus protocol',
        'dqn_agent': 'Deep Q-Network agent for decision making',
        'simulation': 'Main simulation environment coordinating the components'
    }
    
    # Lưu vào tệp JSON
    context_data = {
        'project_structure': code_structure,
        'key_components': relationships,
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs('code_analysis', exist_ok=True)
    with open('code_analysis/code_context.json', 'w') as f:
        json.dump(context_data, f, indent=2)
    
    print("Đã lưu báo cáo cấu trúc code vào code_analysis/code_context.json")
    return context_data

def export_ai_friendly_data(run_id=None):
    """Xuất dữ liệu ở định dạng thân thiện cho AI"""
    print("Đang chuẩn bị dữ liệu định dạng AI...")
    
    # Thư mục chứa dữ liệu phân tích
    data_dir = 'performance_analysis'
    
    # Nếu có ID cụ thể, sử dụng nó, nếu không, lấy tệp gần đây nhất
    if run_id is None:
        # Tìm tệp summary_stats gần đây nhất
        summary_files = sorted(glob.glob(f'{data_dir}/summary_stats_*.json'), 
                               key=os.path.getmtime, reverse=True)
        if not summary_files:
            print("Không tìm thấy dữ liệu phân tích. Hãy chạy mô phỏng trước.")
            return {}
        
        # Lấy ID từ tên tệp (summary_stats_TIMESTAMP.json)
        run_id = os.path.basename(summary_files[0]).split('_')[-1].split('.')[0]
    
    summary_file = f'{data_dir}/summary_stats_{run_id}.json'
    metrics_file = f'{data_dir}/metrics_history_{run_id}.json'
    
    # Kiểm tra xem các tệp có tồn tại không
    if not os.path.exists(summary_file) or not os.path.exists(metrics_file):
        print(f"Không tìm thấy dữ liệu cho run_id: {run_id}")
        return {}
    
    # Đọc dữ liệu
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    # Tính toán thống kê bổ sung cho AI
    ai_stats = {}
    
    # Thêm xu hướng thông lượng
    if 'throughput' in metrics_data:
        throughput_values = metrics_data['throughput']
        ai_stats['throughput_trend'] = {
            'min': min(throughput_values),
            'max': max(throughput_values),
            'avg': sum(throughput_values) / len(throughput_values),
            'start_vs_end': throughput_values[0] - throughput_values[-1] if throughput_values else 0,
            'values': throughput_values
        }
    
    # Thêm xu hướng độ trễ
    if 'latency' in metrics_data:
        latency_values = metrics_data['latency']
        ai_stats['latency_trend'] = {
            'min': min(latency_values),
            'max': max(latency_values),
            'avg': sum(latency_values) / len(latency_values),
            'start_vs_end': latency_values[0] - latency_values[-1] if latency_values else 0,
            'values': latency_values
        }
    
    # Thêm mối tương quan (nếu có đủ dữ liệu)
    if 'congestion' in metrics_data and 'throughput' in metrics_data:
        congestion = metrics_data['congestion']
        throughput = metrics_data['throughput']
        
        # Tính toán mối tương quan đơn giản (chỉ tính khi danh sách có cùng độ dài)
        if len(congestion) == len(throughput) and len(congestion) > 0:
            avg_congestion = sum(congestion) / len(congestion)
            avg_throughput = sum(throughput) / len(throughput)
            
            # Tương quan thô
            correlation = sum((c - avg_congestion) * (t - avg_throughput) 
                            for c, t in zip(congestion, throughput))
            
            ai_stats['congestion_throughput_correlation'] = {
                'value': correlation,
                'interpretation': 'Positive (congestion increases with throughput)' if correlation > 0 
                                else 'Negative (congestion decreases with throughput)'
            }
    
    # Tính toán hiệu quả của ACSC vs MAD-RAPID
    if 'module_stats' in summary_data:
        modules = summary_data['module_stats']
        if 'mad_rapid' in modules and 'acsc' in modules:
            mad_rapid = modules['mad_rapid']
            acsc = modules['acsc']
            
            mad_success_rate = mad_rapid['optimized_tx_count'] / mad_rapid['total_tx_processed'] if mad_rapid['total_tx_processed'] > 0 else 0
            acsc_success_rate = acsc['successful_tx_count'] / acsc['total_tx_processed'] if acsc['total_tx_processed'] > 0 else 0
            
            ai_stats['module_comparison'] = {
                'mad_rapid_success_rate': mad_success_rate,
                'acsc_success_rate': acsc_success_rate,
                'difference': mad_success_rate - acsc_success_rate,
                'recommendation': 'Improve ACSC' if mad_success_rate > acsc_success_rate else 
                                'Improve MAD-RAPID' if acsc_success_rate > mad_success_rate else
                                'Both modules need improvement'
            }
    
    # Tổng hợp báo cáo AI-friendly
    ai_friendly_data = {
        'summary': summary_data,
        'detailed_metrics': metrics_data,
        'ai_analysis': ai_stats,
        'recommendations': {
            'throughput': 'Throughput needs improvement' if ai_stats.get('throughput_trend', {}).get('avg', 0) < 2.0 else 'Throughput is acceptable',
            'latency': 'Latency needs improvement' if ai_stats.get('latency_trend', {}).get('avg', 0) > 100.0 else 'Latency is acceptable',
            'energy': 'Energy consumption is too high' if summary_data.get('performance_stats', {}).get('energy_consumption', 0) > 100000 else 'Energy consumption is acceptable',
            'module_focus': ai_stats.get('module_comparison', {}).get('recommendation', 'No recommendation'),
            'simulation_parameters': 'Consider increasing number of shards' if summary_data.get('simulation_config', {}).get('num_shards', 0) < 8 else 'Shard count is sufficient',
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Lưu dữ liệu
    output_file = f'{data_dir}/ai_friendly_data_{run_id}.json'
    with open(output_file, 'w') as f:
        json.dump(ai_friendly_data, f, indent=2)
    
    print(f"Đã xuất dữ liệu định dạng AI vào {output_file}")
    return ai_friendly_data

def main():
    parser = argparse.ArgumentParser(description='Công cụ phân tích blockchain.')
    
    # Các tùy chọn chạy mô phỏng
    parser.add_argument('--num_steps', type=int, default=100, help='Số lượng bước mô phỏng')
    parser.add_argument('--tx_per_step', type=int, default=20, help='Số lượng giao dịch mỗi bước')
    parser.add_argument('--seed', type=int, default=42, help='Giá trị seed cho random')
    parser.add_argument('--num_shards', type=int, default=4, help='Số lượng shard')
    parser.add_argument('--save_stats', action='store_true', help='Lưu thống kê vào tệp')
    
    # Tùy chọn phân tích
    parser.add_argument('--compare_runs', action='store_true', help='So sánh các lần chạy')
    parser.add_argument('--run_id', type=str, help='ID cụ thể của lần chạy để phân tích')
    parser.add_argument('--visualize', action='store_true', help='Tạo biểu đồ trực quan hóa')
    
    # Tùy chọn mới: xuất dữ liệu cho AI và tạo context code
    parser.add_argument('--export_ai_data', action='store_true', help='Xuất dữ liệu ở định dạng thân thiện với AI')
    parser.add_argument('--create_code_context', action='store_true', help='Tạo báo cáo về cấu trúc và context của mã nguồn')
    
    args = parser.parse_args()
    
    # Tạo báo cáo cấu trúc code nếu được yêu cầu
    if args.create_code_context:
        create_code_context()
        return
    
    # Xuất dữ liệu định dạng AI nếu được yêu cầu
    if args.export_ai_data:
        export_ai_friendly_data(args.run_id)
        return
    
    # So sánh các lần chạy
    if args.compare_runs:
        analyzer = PerformanceAnalyzer()
        analyzer.compare_latest_runs()
        
        if args.visualize and HAS_MATPLOTLIB:
            analyzer.visualize_comparison()
        return
    
    # Chạy mô phỏng mới
    print(f"Chạy mô phỏng với {args.num_steps} bước, {args.tx_per_step} giao dịch mỗi bước")
    simulation = AdvancedBlockchainSimulation(
        num_shards=args.num_shards,
        seed=args.seed
    )
    
    simulation.run(
        num_steps=args.num_steps,
        tx_per_step=args.tx_per_step,
        save_stats=args.save_stats
    )
    
    # Nếu yêu cầu, xuất dữ liệu định dạng AI sau khi chạy mô phỏng
    if args.save_stats:
        # Lấy ID từ lần chạy gần nhất
        summary_files = sorted(glob.glob('performance_analysis/summary_stats_*.json'), 
                              key=os.path.getmtime, reverse=True)
        if summary_files:
            run_id = os.path.basename(summary_files[0]).split('_')[-1].split('.')[0]
            export_ai_friendly_data(run_id)

if __name__ == "__main__":
    main() 