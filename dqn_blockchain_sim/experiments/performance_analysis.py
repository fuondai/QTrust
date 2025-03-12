"""
Script phân tích hiệu suất chi tiết của mô phỏng blockchain
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_blockchain_sim.simulation.advanced_simulation import AdvancedSimulation


class PerformanceAnalyzer:
    """
    Lớp phân tích hiệu suất chi tiết của mô phỏng blockchain
    """
    
    def __init__(self, output_dir: str = "performance_analysis"):
        """
        Khởi tạo analyzer
        
        Args:
            output_dir: Thư mục lưu kết quả
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Kết quả phân tích
        self.results = {}
        self.transaction_data = []
        self.network_data = []
        self.energy_data = []
    
    def run_simulation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chạy mô phỏng với cấu hình cho trước
        
        Args:
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả mô phỏng
        """
        # Tạo mô phỏng
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=config.get('use_real_data', False),
            use_dqn=config.get('use_dqn', True),
            eth_api_key=config.get('eth_api_key', None),
            data_dir=config.get('data_dir', "data"),
            log_dir=self.output_dir
        )
        
        # Chạy mô phỏng
        print(f"Running simulation with {config.get('num_shards', 8)} shards...")
        results = simulation.run_simulation(
            num_steps=config.get('num_steps', 100),
            tx_per_step=config.get('tx_per_step', 100),
            visualize=False,
            save_stats=True
        )
        
        # Thu thập dữ liệu chi tiết
        self._collect_transaction_data(simulation)
        self._collect_network_data(simulation)
        self._collect_energy_data(simulation)
        
        return results
    
    def _collect_transaction_data(self, simulation: AdvancedSimulation) -> None:
        """
        Thu thập dữ liệu chi tiết về giao dịch
        
        Args:
            simulation: Đối tượng mô phỏng
        """
        # Thu thập dữ liệu từ các shard
        for shard_id, shard in simulation.network.shards.items():
            # Giao dịch đã xử lý
            for tx_id, tx in getattr(shard, 'processed_transactions', {}).items():
                source_shard = getattr(tx, 'source_shard', shard_id)
                target_shard = getattr(tx, 'target_shard', shard_id)
                # Xác định giao dịch xuyên mảnh bằng cách so sánh source_shard và target_shard
                is_cross_shard = source_shard != target_shard
                
                tx_data = {
                    'transaction_id': tx_id,
                    'shard_id': shard_id,
                    'status': 'processed',
                    'is_cross_shard': is_cross_shard,
                    'latency': getattr(tx, 'get_latency', lambda: 0)() if callable(getattr(tx, 'get_latency', None)) else 0,
                    'value': getattr(tx, 'amount', 0),
                    'gas_price': getattr(tx, 'gas_price', 0),
                    'gas_limit': getattr(tx, 'gas_limit', 0),
                    'source_shard': source_shard,
                    'target_shard': target_shard
                }
                self.transaction_data.append(tx_data)
                
            # Giao dịch thất bại
            for tx_id, tx in getattr(shard, 'failed_transactions', {}).items():
                source_shard = getattr(tx, 'source_shard', shard_id)
                target_shard = getattr(tx, 'target_shard', shard_id)
                # Xác định giao dịch xuyên mảnh bằng cách so sánh source_shard và target_shard
                is_cross_shard = source_shard != target_shard
                
                tx_data = {
                    'transaction_id': tx_id,
                    'shard_id': shard_id,
                    'status': 'failed',
                    'is_cross_shard': is_cross_shard,
                    'latency': getattr(tx, 'get_latency', lambda: 0)() if callable(getattr(tx, 'get_latency', None)) else 0,
                    'value': getattr(tx, 'amount', 0),
                    'gas_price': getattr(tx, 'gas_price', 0),
                    'gas_limit': getattr(tx, 'gas_limit', 0),
                    'source_shard': source_shard,
                    'target_shard': target_shard
                }
                self.transaction_data.append(tx_data)
    
    def _collect_network_data(self, simulation: AdvancedSimulation) -> None:
        """
        Thu thập dữ liệu chi tiết về mạng
        
        Args:
            simulation: Đối tượng mô phỏng
        """
        # Thu thập dữ liệu từ các shard
        for shard_id, shard in simulation.network.shards.items():
            shard_data = {
                'shard_id': shard_id,
                'num_nodes': len(getattr(shard, 'nodes', [])),
                'throughput': getattr(shard, 'throughput', 0),
                'congestion_level': getattr(shard, 'congestion_level', 0),
                'avg_latency': getattr(shard, 'avg_latency', 0),
                'energy_consumption': getattr(shard, 'energy_consumption', 0),
                'num_processed_tx': len(getattr(shard, 'processed_transactions', {})),
                'num_failed_tx': len(getattr(shard, 'failed_transactions', {}))
            }
            self.network_data.append(shard_data)
    
    def _collect_energy_data(self, simulation: AdvancedSimulation) -> None:
        """
        Thu thập dữ liệu chi tiết về tiêu thụ năng lượng
        
        Args:
            simulation: Đối tượng mô phỏng
        """
        # Thu thập dữ liệu từ các module
        if hasattr(simulation, 'acsc') and hasattr(simulation.acsc, 'get_statistics'):
            acsc_stats = simulation.acsc.get_statistics()
            
            # Phân tích tiêu thụ năng lượng theo loại đồng thuận
            if 'strategy_usage' in acsc_stats:
                for strategy, count in acsc_stats['strategy_usage'].items():
                    # Ước tính năng lượng tiêu thụ cho mỗi loại đồng thuận
                    energy_per_tx = 0
                    if strategy == 'FastBFTConsensus':
                        energy_per_tx = 0.1  # Giả định
                    elif strategy == 'StandardPBFTConsensus':
                        energy_per_tx = 0.5  # Giả định
                    elif strategy == 'RobustBFTConsensus':
                        energy_per_tx = 1.0  # Giả định
                        
                    energy_data = {
                        'component': 'acsc',
                        'strategy': strategy,
                        'count': count,
                        'energy_per_tx': energy_per_tx,
                        'total_energy': energy_per_tx * count
                    }
                    self.energy_data.append(energy_data)
    
    def analyze_transaction_success_by_type(self) -> pd.DataFrame:
        """
        Phân tích tỉ lệ thành công theo loại giao dịch
        
        Returns:
            DataFrame kết quả phân tích
        """
        if not self.transaction_data:
            print("No transaction data to analyze.")
            return pd.DataFrame()
            
        # Tạo DataFrame
        df = pd.DataFrame(self.transaction_data)
        
        # Kiểm tra trạng thái giao dịch
        print(f"Các trạng thái giao dịch duy nhất: {df['status'].unique()}")
        
        # Phân loại giao dịch theo giá trị
        df['value_category'] = pd.cut(
            df['value'],
            bins=[0, 10, 50, 100, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Phân tích tỉ lệ thành công theo loại giao dịch
        # Kiểm tra và xác định các trạng thái thành công (có thể là 'processed', 'confirmed', 'success', v.v.)
        success_states = ['processed', 'confirmed', 'success']
        df['is_success'] = df['status'].apply(lambda x: any(state in str(x).lower() for state in success_states))
        
        success_by_type = df.groupby(['is_cross_shard', 'value_category']).agg(
            total_count=('transaction_id', 'count'),
            success_count=('is_success', 'sum')
        )
        
        # Tính tỉ lệ thành công
        success_by_type['success_rate'] = success_by_type['success_count'] / success_by_type['total_count']
        
        return success_by_type.reset_index()
    
    def analyze_performance_by_network_size(self) -> pd.DataFrame:
        """
        Đánh giá hiệu suất theo kích thước mạng
        
        Returns:
            DataFrame kết quả phân tích
        """
        if not self.network_data:
            print("No network data to analyze.")
            return pd.DataFrame()
            
        # Tạo DataFrame
        df = pd.DataFrame(self.network_data)
        
        # Tính tỉ lệ thành công cho mỗi shard
        df['success_rate'] = df['num_processed_tx'] / (df['num_processed_tx'] + df['num_failed_tx']).replace(0, 1)
        
        # Tính hiệu suất trung bình theo số lượng node
        performance_by_size = df.groupby('num_nodes').agg(
            avg_throughput=('throughput', 'mean'),
            avg_latency=('avg_latency', 'mean'),
            avg_congestion=('congestion_level', 'mean'),
            avg_success_rate=('success_rate', 'mean'),
            avg_energy=('energy_consumption', 'mean')
        )
        
        return performance_by_size.reset_index()
    
    def analyze_energy_consumption(self) -> pd.DataFrame:
        """
        Phân tích tiêu thụ năng lượng
        
        Returns:
            DataFrame kết quả phân tích
        """
        if not self.energy_data:
            print("No energy data to analyze.")
            # Tạo dữ liệu năng lượng mặc định nếu không có
            default_energy = [
                {'component': 'blockchain', 'strategy': 'default', 'count': 0, 'energy_per_tx': 0, 'total_energy': 0}
            ]
            if 'acsc' in getattr(self, 'simulation', {}) and self.simulation.acsc:
                acsc_stats = self.simulation.acsc.get_statistics()
                tx_count = acsc_stats.get('total_transactions', 0)
                if tx_count > 0:
                    energy_per_tx = 27.0  # Giá trị mặc định từ các thống kê đã thấy
                    default_energy.append({
                        'component': 'acsc',
                        'strategy': 'combined',
                        'count': tx_count,
                        'energy_per_tx': energy_per_tx,
                        'total_energy': energy_per_tx * tx_count
                    })
            return pd.DataFrame(default_energy)
            
        # Tạo DataFrame
        df = pd.DataFrame(self.energy_data)
        
        # Kiểm tra dữ liệu
        print(f"Dữ liệu năng lượng: {len(df)} dòng")
        if not df.empty:
            print(f"Tổng năng lượng: {df['total_energy'].sum()}")
        
        # Tính tổng năng lượng tiêu thụ theo thành phần
        energy_by_component = df.groupby('component').agg(
            total_energy=('total_energy', 'sum')
        )
        
        return energy_by_component.reset_index()
    
    def run_analysis(self, consensus_configs, save_plots=True):
        """
        Chạy phân tích hiệu năng cho các cấu hình khác nhau
        
        Args:
            consensus_configs: Danh sách cấu hình đồng thuận
            save_plots: Có lưu biểu đồ hay không
        """
        all_results = []
        
        for i, config in enumerate(consensus_configs):
            print(f"Running analysis {i+1}/{len(consensus_configs)}...")
            
            # Khởi tạo mô phỏng
            sim_config = {
                "num_shards": config.get("num_shards", 4),
                "use_dqn": config.get("use_dqn", True),
                "consensus_protocol": config.get("consensus_protocol", "PBFT")
            }
            
            # Chạy mô phỏng và thu thập kết quả
            simulation_results = self.run_simulation(sim_config)
            
            # Phân tích giao dịch theo loại
            tx_analysis = self.analyze_transaction_success_by_type()
            
            # Phân tích hiệu suất theo kích thước mạng
            performance_analysis = self.analyze_performance_by_network_size()
            
            # Phân tích tiêu thụ năng lượng
            energy_analysis = self.analyze_energy_consumption()
            
            # Lưu kết quả
            result = {
                "config_id": i,
                "config": sim_config,
                "tx_analysis": tx_analysis,
                "performance_analysis": performance_analysis,
                "energy_analysis": energy_analysis,
                "summary_stats": simulation_results
            }
            all_results.append(result)
            
            # Lưu thống kê tóm tắt vào file CSV
            self._save_summary_stats_to_csv(all_results)
        
        # Lưu tất cả kết quả
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"All results saved to {self.output_dir}")
        
        # Tạo biểu đồ
        if save_plots:
            self._plot_success_rate_by_tx_type(all_results)
            self._plot_performance_by_network_size(all_results)
            self._plot_energy_consumption(all_results)
        
        return all_results
    
    def _save_summary_stats_to_csv(self, results):
        """
        Lưu thống kê tóm tắt vào file CSV
        
        Args:
            results: Kết quả phân tích
        """
        # Tạo DataFrame từ kết quả
        data = []
        for result in results:
            config = result['config']
            summary = result['summary_stats']['performance_stats']
            
            # Lấy thông tin từ summary_stats
            row = {
                'config_id': result['config_id'],
                'num_shards': config['num_shards'],
                'num_steps': 100,  # Giá trị mặc định
                'tx_per_step': 100,  # Giá trị mặc định
                'use_dqn': 'True' if config['use_dqn'] else 'False',
                'throughput': summary['avg_throughput'],
                'latency': summary['avg_latency'],
                'success_rate': result['summary_stats']['transaction_stats']['success_rate'],
                'energy_consumption': summary['energy_consumption']
            }
            data.append(row)
        
        # Chuyển đổi thành DataFrame và lưu vào CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.output_dir, 'analysis_summary.csv'), index=False)
    
    def _plot_success_rate_by_tx_type(self, results: List[Dict[str, Any]]) -> None:
        """
        Tạo biểu đồ tỉ lệ thành công theo loại giao dịch
        
        Args:
            results: Danh sách kết quả phân tích
        """
        # Kết hợp dữ liệu từ tất cả các cấu hình
        all_tx_data = []
        
        for i, result in enumerate(results):
            if ('transaction_analysis' in result and 
                isinstance(result['transaction_analysis'], pd.DataFrame) and 
                not result['transaction_analysis'].empty):
                tx_analysis = pd.DataFrame(result['transaction_analysis'])
                tx_analysis['config_id'] = i
                all_tx_data.append(tx_analysis)
                
        if not all_tx_data:
            print("No transaction analysis data to plot.")
            return
            
        # Kết hợp dữ liệu
        combined_df = pd.concat(all_tx_data)
        
        # Tạo biểu đồ
        plt.figure(figsize=(12, 8))
        
        # Biểu đồ tỉ lệ thành công theo loại giao dịch
        sns.barplot(x='value_category', y='success_rate', hue='is_cross_shard', data=combined_df)
        
        plt.title('Success Rate by Transaction Type and Value')
        plt.xlabel('Transaction Value Category')
        plt.ylabel('Success Rate')
        plt.legend(title='Cross-Shard', labels=['No', 'Yes'])
        
        # Lưu biểu đồ
        chart_file = os.path.join(self.output_dir, "success_rate_by_tx_type.png")
        plt.savefig(chart_file)
        
        print(f"Chart saved to {chart_file}")
    
    def _plot_performance_by_network_size(self, results: List[Dict[str, Any]]) -> None:
        """
        Tạo biểu đồ hiệu suất theo kích thước mạng
        
        Args:
            results: Danh sách kết quả phân tích
        """
        # Kết hợp dữ liệu từ tất cả các cấu hình
        all_network_data = []
        
        for i, result in enumerate(results):
            if ('performance_analysis' in result and 
                isinstance(result['performance_analysis'], pd.DataFrame) and 
                not result['performance_analysis'].empty):
                network_analysis = pd.DataFrame(result['performance_analysis'])
                network_analysis['config_id'] = i
                all_network_data.append(network_analysis)
                
        if not all_network_data:
            print("No network analysis data to plot.")
            return
            
        # Kết hợp dữ liệu
        combined_df = pd.concat(all_network_data)
        
        # Tạo biểu đồ
        plt.figure(figsize=(15, 10))
        
        # 1. Throughput vs Network Size
        plt.subplot(2, 2, 1)
        sns.lineplot(x='num_nodes', y='avg_throughput', data=combined_df, marker='o')
        plt.title('Throughput vs Network Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Throughput (tx/step)')
        
        # 2. Latency vs Network Size
        plt.subplot(2, 2, 2)
        sns.lineplot(x='num_nodes', y='avg_latency', data=combined_df, marker='o')
        plt.title('Latency vs Network Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Latency (ms)')
        
        # 3. Success Rate vs Network Size
        plt.subplot(2, 2, 3)
        sns.lineplot(x='num_nodes', y='avg_success_rate', data=combined_df, marker='o')
        plt.title('Success Rate vs Network Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Success Rate')
        
        # 4. Energy Consumption vs Network Size
        plt.subplot(2, 2, 4)
        sns.lineplot(x='num_nodes', y='avg_energy', data=combined_df, marker='o')
        plt.title('Energy Consumption vs Network Size')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Average Energy Consumption')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_file = os.path.join(self.output_dir, "performance_by_network_size.png")
        plt.savefig(chart_file)
        
        print(f"Chart saved to {chart_file}")
    
    def _plot_energy_consumption(self, results: List[Dict[str, Any]]) -> None:
        """
        Tạo biểu đồ tiêu thụ năng lượng
        
        Args:
            results: Danh sách kết quả phân tích
        """
        # Kết hợp dữ liệu từ tất cả các cấu hình
        all_energy_data = []
        
        for i, result in enumerate(results):
            if ('energy_analysis' in result and 
                isinstance(result['energy_analysis'], pd.DataFrame) and 
                not result['energy_analysis'].empty):
                energy_analysis = pd.DataFrame(result['energy_analysis'])
                energy_analysis['config_id'] = i
                all_energy_data.append(energy_analysis)
                
        if not all_energy_data:
            print("No energy analysis data to plot.")
            return
            
        # Kết hợp dữ liệu
        combined_df = pd.concat(all_energy_data)
        
        # Tạo biểu đồ
        plt.figure(figsize=(10, 6))
        
        # Biểu đồ tiêu thụ năng lượng theo thành phần
        sns.barplot(x='component', y='total_energy', data=combined_df)
        
        plt.title('Energy Consumption by Component')
        plt.xlabel('Component')
        plt.ylabel('Total Energy Consumption')
        
        # Lưu biểu đồ
        chart_file = os.path.join(self.output_dir, "energy_consumption.png")
        plt.savefig(chart_file)
        
        print(f"Chart saved to {chart_file}")


def main():
    """
    Hàm chính để chạy phân tích hiệu suất
    """
    # Tạo thư mục đầu ra
    output_dir = "performance_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo analyzer
    analyzer = PerformanceAnalyzer(output_dir)
    
    # Tạo các cấu hình mô phỏng
    configs = []
    
    # 1. Cấu hình với số lượng shard khác nhau
    for num_shards in [4, 8, 16]:
        config = {
            'name': f"shard_analysis_{num_shards}",
            'num_shards': num_shards,
            'num_steps': 100,
            'tx_per_step': 100,
            'use_dqn': True
        }
        configs.append(config)
    
    # 2. Cấu hình với khối lượng giao dịch khác nhau
    for tx_per_step in [10, 20, 50]:
        config = {
            'name': f"load_analysis_{tx_per_step}",
            'num_shards': 8,
            'num_steps': 100,
            'tx_per_step': tx_per_step,
            'use_dqn': True
        }
        configs.append(config)
    
    # 3. Cấu hình với và không có DQN
    for use_dqn in [True, False]:
        config = {
            'name': f"dqn_analysis_{'with' if use_dqn else 'without'}",
            'num_shards': 8,
            'num_steps': 100,
            'tx_per_step': 100,
            'use_dqn': use_dqn
        }
        configs.append(config)
    
    # Chạy phân tích
    analyzer.run_analysis(configs)


if __name__ == "__main__":
    main() 