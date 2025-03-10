"""
Script so sánh hiệu suất của các thuật toán đồng thuận khác nhau
"""

import os
import sys
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_blockchain_sim.simulation.advanced_simulation import AdvancedSimulation
from dqn_blockchain_sim.blockchain.consensus import ConsensusProtocol


class ConsensusSimulator:
    """
    Lớp mô phỏng và so sánh các thuật toán đồng thuận
    """
    
    def __init__(self, output_dir: str = "consensus_comparison"):
        """
        Khởi tạo simulator
        
        Args:
            output_dir: Thư mục lưu kết quả
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Các thuật toán đồng thuận được hỗ trợ
        self.consensus_protocols = {
            "pow": self._simulate_pow,
            "pos": self._simulate_pos,
            "pbft": self._simulate_pbft,
            "acsc": self._simulate_acsc
        }
        
        # Kết quả mô phỏng
        self.results = {}
    
    def _simulate_pow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mô phỏng thuật toán đồng thuận Proof of Work
        
        Args:
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả mô phỏng
        """
        # Tạo mô phỏng với PoW
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=False,
            use_dqn=config.get('use_dqn', True),
            log_dir=self.output_dir
        )
        
        # Cấu hình PoW cho tất cả các shard
        for shard_id, shard in simulation.network.shards.items():
            if hasattr(shard, 'validator'):
                shard.validator.consensus_protocol = ConsensusProtocol.POW
                shard.validator.difficulty = config.get('difficulty', 4)  # Độ khó của PoW
        
        # Chạy mô phỏng
        print(f"Running PoW simulation with {config.get('num_shards', 8)} shards...")
        results = simulation.run_simulation(
            num_steps=config.get('num_steps', 100),
            tx_per_step=config.get('tx_per_step', 20),
            visualize=False,
            save_stats=True
        )
        
        # Thêm thông tin về thuật toán đồng thuận
        results['consensus_protocol'] = 'pow'
        results['difficulty'] = config.get('difficulty', 4)
        results['energy_consumption'] = self._calculate_pow_energy(results, config)
        
        return results
    
    def _simulate_pos(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mô phỏng thuật toán đồng thuận Proof of Stake
        
        Args:
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả mô phỏng
        """
        # Tạo mô phỏng với PoS
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=False,
            use_dqn=config.get('use_dqn', True),
            log_dir=self.output_dir
        )
        
        # Cấu hình PoS cho tất cả các shard
        for shard_id, shard in simulation.network.shards.items():
            if hasattr(shard, 'validator'):
                shard.validator.consensus_protocol = ConsensusProtocol.POS
                shard.validator.min_stake = config.get('min_stake', 100)  # Stake tối thiểu
        
        # Chạy mô phỏng
        print(f"Running PoS simulation with {config.get('num_shards', 8)} shards...")
        results = simulation.run_simulation(
            num_steps=config.get('num_steps', 100),
            tx_per_step=config.get('tx_per_step', 20),
            visualize=False,
            save_stats=True
        )
        
        # Thêm thông tin về thuật toán đồng thuận
        results['consensus_protocol'] = 'pos'
        results['min_stake'] = config.get('min_stake', 100)
        results['energy_consumption'] = self._calculate_pos_energy(results, config)
        
        return results
    
    def _simulate_pbft(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mô phỏng thuật toán đồng thuận Practical Byzantine Fault Tolerance
        
        Args:
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả mô phỏng
        """
        # Tạo mô phỏng với PBFT
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=False,
            use_dqn=config.get('use_dqn', True),
            log_dir=self.output_dir
        )
        
        # Cấu hình PBFT cho tất cả các shard
        for shard_id, shard in simulation.network.shards.items():
            if hasattr(shard, 'validator'):
                shard.validator.consensus_protocol = ConsensusProtocol.PBFT
                shard.validator.fault_tolerance = config.get('fault_tolerance', 0.33)  # Khả năng chịu lỗi
        
        # Chạy mô phỏng
        print(f"Running PBFT simulation with {config.get('num_shards', 8)} shards...")
        results = simulation.run_simulation(
            num_steps=config.get('num_steps', 100),
            tx_per_step=config.get('tx_per_step', 20),
            visualize=False,
            save_stats=True
        )
        
        # Thêm thông tin về thuật toán đồng thuận
        results['consensus_protocol'] = 'pbft'
        results['fault_tolerance'] = config.get('fault_tolerance', 0.33)
        results['energy_consumption'] = self._calculate_pbft_energy(results, config)
        
        return results
    
    def _simulate_acsc(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mô phỏng thuật toán đồng thuận Adaptive Cross-Shard Consensus
        
        Args:
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả mô phỏng
        """
        # Tạo mô phỏng với ACSC
        simulation = AdvancedSimulation(
            num_shards=config.get('num_shards', 8),
            use_real_data=False,
            use_dqn=config.get('use_dqn', True),
            log_dir=self.output_dir
        )
        
        # ACSC đã được tích hợp sẵn trong AdvancedSimulation
        
        # Chạy mô phỏng
        print(f"Running ACSC simulation with {config.get('num_shards', 8)} shards...")
        results = simulation.run_simulation(
            num_steps=config.get('num_steps', 100),
            tx_per_step=config.get('tx_per_step', 20),
            visualize=False,
            save_stats=True
        )
        
        # Thêm thông tin về thuật toán đồng thuận
        results['consensus_protocol'] = 'acsc'
        
        return results
    
    def _calculate_pow_energy(self, results: Dict[str, Any], config: Dict[str, Any]) -> float:
        """
        Tính toán năng lượng tiêu thụ cho PoW
        
        Args:
            results: Kết quả mô phỏng
            config: Cấu hình mô phỏng
            
        Returns:
            Năng lượng tiêu thụ
        """
        # Mô hình năng lượng đơn giản cho PoW: Năng lượng tỉ lệ với độ khó và số lượng giao dịch
        difficulty = config.get('difficulty', 4)
        num_transactions = results.get('total_transactions', 0)
        
        # Hệ số năng lượng cho PoW (đơn vị tùy ý)
        energy_factor = 10.0
        
        return energy_factor * difficulty * num_transactions
    
    def _calculate_pos_energy(self, results: Dict[str, Any], config: Dict[str, Any]) -> float:
        """
        Tính toán năng lượng tiêu thụ cho PoS
        
        Args:
            results: Kết quả mô phỏng
            config: Cấu hình mô phỏng
            
        Returns:
            Năng lượng tiêu thụ
        """
        # Mô hình năng lượng đơn giản cho PoS: Năng lượng thấp hơn nhiều so với PoW
        num_transactions = results.get('total_transactions', 0)
        num_shards = config.get('num_shards', 8)
        
        # Hệ số năng lượng cho PoS (đơn vị tùy ý)
        energy_factor = 0.1
        
        return energy_factor * num_transactions * num_shards
    
    def _calculate_pbft_energy(self, results: Dict[str, Any], config: Dict[str, Any]) -> float:
        """
        Tính toán năng lượng tiêu thụ cho PBFT
        
        Args:
            results: Kết quả mô phỏng
            config: Cấu hình mô phỏng
            
        Returns:
            Năng lượng tiêu thụ
        """
        # Mô hình năng lượng đơn giản cho PBFT: Năng lượng phụ thuộc vào số lượng giao dịch và số lượng shard
        num_transactions = results.get('total_transactions', 0)
        num_shards = config.get('num_shards', 8)
        
        # Hệ số năng lượng cho PBFT (đơn vị tùy ý)
        energy_factor = 0.5
        
        return energy_factor * num_transactions * num_shards
    
    def run_comparison(self, protocols: List[str] = None, config: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        Chạy so sánh các thuật toán đồng thuận
        
        Args:
            protocols: Danh sách các thuật toán đồng thuận cần so sánh
            config: Cấu hình mô phỏng
            
        Returns:
            Kết quả so sánh
        """
        if protocols is None:
            protocols = list(self.consensus_protocols.keys())
            
        if config is None:
            config = {
                'num_shards': 8,
                'num_steps': 100,
                'tx_per_step': 20,
                'use_dqn': True,
                'difficulty': 4,  # Cho PoW
                'min_stake': 100,  # Cho PoS
                'fault_tolerance': 0.33  # Cho PBFT
            }
            
        # Chạy mô phỏng cho từng thuật toán
        for protocol in protocols:
            if protocol in self.consensus_protocols:
                print(f"Simulating {protocol.upper()} consensus protocol...")
                self.results[protocol] = self.consensus_protocols[protocol](config)
                
        # Lưu kết quả
        self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """
        Lưu kết quả so sánh
        """
        # Lưu kết quả chi tiết
        results_file = os.path.join(self.output_dir, f"consensus_comparison_{int(time.time())}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Tạo DataFrame tóm tắt
        summary_data = []
        for protocol, results in self.results.items():
            row = {
                'protocol': protocol,
                'throughput': results.get('avg_throughput', 0),
                'latency': results.get('avg_latency', 0),
                'success_rate': results.get('success_rate', 0),
                'energy_consumption': results.get('energy_consumption', 0)
            }
            summary_data.append(row)
            
        # Lưu tóm tắt
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(self.output_dir, f"consensus_summary_{int(time.time())}.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Results saved to {results_file} and {summary_file}")
    
    def visualize_comparison(self) -> None:
        """
        Hiển thị kết quả so sánh dưới dạng biểu đồ
        """
        if not self.results:
            print("No results to visualize. Run comparison first.")
            return
            
        # Tạo DataFrame từ kết quả
        data = []
        for protocol, results in self.results.items():
            row = {
                'protocol': protocol,
                'throughput': results.get('avg_throughput', 0),
                'latency': results.get('avg_latency', 0),
                'success_rate': results.get('success_rate', 0),
                'energy_consumption': results.get('energy_consumption', 0)
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Tạo biểu đồ
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Throughput
        axs[0, 0].bar(df['protocol'], df['throughput'])
        axs[0, 0].set_title('Throughput by Consensus Protocol')
        axs[0, 0].set_xlabel('Protocol')
        axs[0, 0].set_ylabel('Throughput (tx/step)')
        
        # 2. Latency
        axs[0, 1].bar(df['protocol'], df['latency'])
        axs[0, 1].set_title('Latency by Consensus Protocol')
        axs[0, 1].set_xlabel('Protocol')
        axs[0, 1].set_ylabel('Latency (ms)')
        
        # 3. Success Rate
        axs[1, 0].bar(df['protocol'], df['success_rate'])
        axs[1, 0].set_title('Success Rate by Consensus Protocol')
        axs[1, 0].set_xlabel('Protocol')
        axs[1, 0].set_ylabel('Success Rate')
        
        # 4. Energy Consumption
        axs[1, 1].bar(df['protocol'], df['energy_consumption'])
        axs[1, 1].set_title('Energy Consumption by Consensus Protocol')
        axs[1, 1].set_xlabel('Protocol')
        axs[1, 1].set_ylabel('Energy Consumption')
        
        plt.tight_layout()
        
        # Lưu biểu đồ
        chart_file = os.path.join(self.output_dir, f"consensus_comparison_{int(time.time())}.png")
        plt.savefig(chart_file)
        
        # Hiển thị biểu đồ
        plt.show()
        
        print(f"Chart saved to {chart_file}")


def main():
    """
    Hàm chính để chạy so sánh các thuật toán đồng thuận
    """
    # Tạo thư mục đầu ra
    output_dir = "consensus_comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Khởi tạo simulator
    simulator = ConsensusSimulator(output_dir)
    
    # Cấu hình mô phỏng
    config = {
        'num_shards': 8,
        'num_steps': 100,
        'tx_per_step': 20,
        'use_dqn': True,
        'difficulty': 4,  # Cho PoW
        'min_stake': 100,  # Cho PoS
        'fault_tolerance': 0.33  # Cho PBFT
    }
    
    # Chạy so sánh
    simulator.run_comparison(['pow', 'pos', 'pbft', 'acsc'], config)
    
    # Hiển thị kết quả
    simulator.visualize_comparison()


if __name__ == "__main__":
    main() 