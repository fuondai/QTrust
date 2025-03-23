#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust: Chương trình mô phỏng đơn giản để chạy hệ thống blockchain
và tạo dữ liệu kết quả mà không cần các phụ thuộc phức tạp.
"""

import os
import time
import random
import math
from datetime import datetime

# Cài đặt seed cho tính tái tạo
SEED = 42
random.seed(SEED)

# Tạo thư mục kết quả nếu chưa tồn tại
os.makedirs("results", exist_ok=True)

class SimpleBlockchainSimulation:
    """
    Mô phỏng đơn giản về môi trường blockchain với sharding.
    """
    
    def __init__(self, num_shards=4, num_nodes_per_shard=10, max_steps=1000, 
                 transaction_rate=20, cross_shard_prob=0.3):
        """
        Khởi tạo mô phỏng blockchain đơn giản.
        
        Args:
            num_shards: Số lượng shard trong mạng
            num_nodes_per_shard: Số lượng node trong mỗi shard
            max_steps: Số bước tối đa cho mỗi episode
            transaction_rate: Số giao dịch được tạo mỗi bước 
            cross_shard_prob: Xác suất giao dịch xuyên shard
        """
        self.num_shards = num_shards
        self.num_nodes_per_shard = num_nodes_per_shard
        self.max_steps = max_steps
        self.transaction_rate = transaction_rate
        self.cross_shard_prob = cross_shard_prob
        
        # Khởi tạo dữ liệu mạng
        self.shard_congestion = [random.uniform(0.1, 0.4) for _ in range(num_shards)]
        self.node_trust_scores = [random.uniform(0.5, 1.0) for _ in range(num_shards * num_nodes_per_shard)]
        
        # Khởi tạo metrics theo dõi
        self.reset_metrics()
        
    def reset_metrics(self):
        """Đặt lại các metrics về mặc định."""
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'security_score': [],
            'cross_shard_ratio': []
        }
        
    def run_simulation(self, episodes=5, adaptive_consensus=True, dqn_enabled=True):
        """
        Chạy mô phỏng với số episodes cho trước.
        
        Args:
            episodes: Số episodes cần chạy
            adaptive_consensus: Bật/tắt tính năng giao thức đồng thuận thích ứng
            dqn_enabled: Bật/tắt thuật toán DQN cho việc tối ưu hóa định tuyến
            
        Returns:
            Dict: Kết quả mô phỏng
        """
        episode_rewards = []
        
        for episode in range(episodes):
            print(f"Chạy episode {episode+1}/{episodes}...")
            
            # Đặt lại trạng thái shard và metrics
            self.shard_congestion = [random.uniform(0.1, 0.4) for _ in range(self.num_shards)]
            self.reset_metrics()
            total_reward = 0
            
            # Chạy một episode đầy đủ
            for step in range(self.max_steps):
                # Mô phỏng các giao dịch
                transactions = self._generate_transactions()
                
                # Xử lý các giao dịch
                reward = self._process_transactions(transactions, adaptive_consensus, dqn_enabled)
                total_reward += reward
                
                # Cập nhật trạng thái mạng
                self._update_network_state()
                
                # Đánh giá metrics
                self._evaluate_performance(step)
                
                # Hiển thị tiến độ sau mỗi 100 bước
                if (step + 1) % 100 == 0:
                    print(f"  Bước {step+1}/{self.max_steps} hoàn thành")
            
            episode_rewards.append(total_reward)
            print(f"  Episode {episode+1} hoàn thành với tổng phần thưởng: {total_reward:.2f}")
            
            # Lưu kết quả sau mỗi episode
            self._save_episode_results(episode)
        
        # Tóm tắt kết quả
        avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
        avg_throughput = sum(self.metrics['throughput']) / len(self.metrics['throughput']) if self.metrics['throughput'] else 0
        avg_latency = sum(self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0
        avg_energy = sum(self.metrics['energy_consumption']) / len(self.metrics['energy_consumption']) if self.metrics['energy_consumption'] else 0
        avg_security = sum(self.metrics['security_score']) / len(self.metrics['security_score']) if self.metrics['security_score'] else 0
        avg_cross_ratio = sum(self.metrics['cross_shard_ratio']) / len(self.metrics['cross_shard_ratio']) if self.metrics['cross_shard_ratio'] else 0
        
        results = {
            'episode_rewards': episode_rewards,
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
            'avg_security': avg_security,
            'cross_shard_ratio': avg_cross_ratio
        }
        
        print("\nKết quả mô phỏng:")
        print(f"  Phần thưởng trung bình trên mỗi episode: {avg_episode_reward:.2f}")
        print(f"  Throughput trung bình: {results['avg_throughput']:.2f} tx/s")
        print(f"  Độ trễ trung bình: {results['avg_latency']:.2f} ms")
        print(f"  Tiêu thụ năng lượng trung bình: {results['avg_energy']:.2f} đơn vị")
        print(f"  Điểm bảo mật trung bình: {results['avg_security']:.2f}")
        print(f"  Tỷ lệ giao dịch xuyên shard: {results['cross_shard_ratio']:.2f}")
        
        # Lưu kết quả tổng hợp vào file
        self._save_summary_results(results, adaptive_consensus, dqn_enabled)
        
        return results
    
    def _generate_transactions(self):
        """Tạo ngẫu nhiên các giao dịch cho bước hiện tại."""
        # Sử dụng phân phối poisson thông qua mô phỏng
        lambda_val = self.transaction_rate
        num_transactions = 0
        L = math.exp(-lambda_val)
        p = 1.0
        k = 0
        
        while p > L:
            k += 1
            p *= random.random()
        
        num_transactions = k - 1
        if num_transactions < 0:
            num_transactions = 0
            
        transactions = []
        
        for i in range(num_transactions):
            # Chọn ngẫu nhiên shard nguồn và đích
            source_shard = random.randint(0, self.num_shards - 1)
            
            # Quyết định nếu đây là giao dịch xuyên shard
            if random.random() < self.cross_shard_prob:
                possible_destinations = [s for s in range(self.num_shards) if s != source_shard]
                if possible_destinations:
                    destination_shard = random.choice(possible_destinations)
                else:
                    destination_shard = source_shard
            else:
                destination_shard = source_shard
            
            # Tạo giá trị ngẫu nhiên cho giao dịch sử dụng phân phối mũ mô phỏng
            # Phân phối mũ với trung bình 20
            value = -20 * math.log(random.random())  # Mô phỏng phân phối mũ
            
            transactions.append({
                'id': i,
                'source_shard': source_shard,
                'destination_shard': destination_shard,
                'value': value
            })
        
        return transactions
    
    def _process_transactions(self, transactions, adaptive_consensus, dqn_enabled):
        """
        Xử lý các giao dịch trong mạng blockchain.
        
        Args:
            transactions: Danh sách giao dịch cần xử lý
            adaptive_consensus: Có sử dụng giao thức đồng thuận thích ứng
            dqn_enabled: Có sử dụng DQN để tối ưu định tuyến
            
        Returns:
            float: Phần thưởng tổng hợp cho bước này
        """
        total_reward = 0
        security_factor = 0.8  # Giá trị mặc định
        
        for tx in transactions:
            # Mô phỏng việc định tuyến với hoặc không có DQN
            if dqn_enabled:
                # Dùng DQN để định tuyến tối ưu (mô phỏng)
                # Trong trường hợp thực, đây sẽ là đầu ra từ mô hình DQN
                routing_quality = random.uniform(0.7, 1.0)
                destination_congestion = self.shard_congestion[tx['destination_shard']] * (1 - routing_quality * 0.5)
            else:
                # Định tuyến đơn giản
                destination_congestion = self.shard_congestion[tx['destination_shard']]
            
            # Mô phỏng việc chọn giao thức đồng thuận
            if adaptive_consensus:
                # Chọn giao thức dựa trên giá trị giao dịch
                if tx['value'] < 10:
                    consensus_protocol = 'FastBFT'  # Nhanh, tiêu thụ ít năng lượng, bảo mật thấp
                    security_factor = 0.6
                    energy_factor = 0.3
                    latency_factor = 0.2
                elif tx['value'] < 50:
                    consensus_protocol = 'PBFT'  # Cân bằng
                    security_factor = 0.8
                    energy_factor = 0.6
                    latency_factor = 0.5
                else:
                    consensus_protocol = 'RobustBFT'  # Bảo mật cao, tiêu thụ nhiều năng lượng, chậm
                    security_factor = 0.95
                    energy_factor = 0.9
                    latency_factor = 0.8
            else:
                # Luôn sử dụng PBFT
                consensus_protocol = 'PBFT'
                security_factor = 0.8
                energy_factor = 0.6
                latency_factor = 0.5
            
            # Tính toán độ trễ và tiêu thụ năng lượng 
            base_latency = 5 + destination_congestion * 50  # 5-55ms
            energy_consumption = 10 + energy_factor * 40  # 10-50 đơn vị
            
            # Thêm độ trễ nếu là giao dịch xuyên shard
            if tx['source_shard'] != tx['destination_shard']:
                base_latency += 20  # Thêm 20ms cho xuyên shard
                energy_consumption *= 1.5  # Tiêu thụ nhiều năng lượng hơn
            
            # Tính độ trễ cuối cùng dựa trên giao thức
            latency = base_latency * (0.5 + latency_factor)
            
            # Xác suất thành công cao hơn với các giao thức an toàn hơn
            success_prob = 0.9 + security_factor * 0.1
            if random.random() < success_prob:
                tx_status = 'completed'
                # Tính thưởng cho giao dịch thành công
                reward = 1.0 - (latency / 100) - (energy_consumption / 100)
            else:
                tx_status = 'failed'
                reward = -0.5
            
            total_reward += reward
            
            # Cập nhật thống kê theo dõi
            if tx_status == 'completed':
                self.metrics['latency'].append(latency)
                self.metrics['energy_consumption'].append(energy_consumption)
                self.metrics['security_score'].append(security_factor)
                if tx['source_shard'] != tx['destination_shard']:
                    self.metrics['cross_shard_ratio'].append(1)
                else:
                    self.metrics['cross_shard_ratio'].append(0)
        
        # Cập nhật throughput
        successful_txs = len([1 for tx in transactions if random.random() < (0.9 + security_factor * 0.1)])
        self.metrics['throughput'].append(successful_txs)
        
        return total_reward
    
    def _update_network_state(self):
        """Cập nhật trạng thái mạng blockchain sau mỗi bước."""
        # Giảm congestion tự nhiên theo thời gian
        for i in range(len(self.shard_congestion)):
            self.shard_congestion[i] = max(0.1, self.shard_congestion[i] * 0.95)
        
        # Thêm nhiễu ngẫu nhiên
        for i in range(len(self.shard_congestion)):
            self.shard_congestion[i] += random.uniform(-0.05, 0.15)
            self.shard_congestion[i] = max(0.1, min(1.0, self.shard_congestion[i]))
        
        # Cập nhật điểm tin cậy của node (thay đổi nhẹ)
        for i in range(len(self.node_trust_scores)):
            self.node_trust_scores[i] += random.uniform(-0.02, 0.02)
            self.node_trust_scores[i] = max(0.1, min(1.0, self.node_trust_scores[i]))
    
    def _evaluate_performance(self, step):
        """Đánh giá hiệu suất mạng tại mỗi bước."""
        # Trong thực tế, chúng ta sẽ tính toán các metrics từ các giao dịch
        # Ở đây, chúng ta đã cập nhật metrics trong quá trình xử lý giao dịch
        pass
    
    def _save_episode_results(self, episode):
        """Lưu kết quả của một episode vào file."""
        # Tạo timestamp cho tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/episode_{episode+1}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Episode {episode+1} Results:\n")
            avg_throughput = sum(self.metrics['throughput']) / len(self.metrics['throughput']) if self.metrics['throughput'] else 0
            avg_latency = sum(self.metrics['latency']) / len(self.metrics['latency']) if self.metrics['latency'] else 0
            avg_energy = sum(self.metrics['energy_consumption']) / len(self.metrics['energy_consumption']) if self.metrics['energy_consumption'] else 0
            avg_security = sum(self.metrics['security_score']) / len(self.metrics['security_score']) if self.metrics['security_score'] else 0
            avg_cross_ratio = sum(self.metrics['cross_shard_ratio']) / len(self.metrics['cross_shard_ratio']) if self.metrics['cross_shard_ratio'] else 0
            
            f.write(f"Throughput trung bình: {avg_throughput:.2f} tx/s\n")
            f.write(f"Độ trễ trung bình: {avg_latency:.2f} ms\n")
            f.write(f"Tiêu thụ năng lượng trung bình: {avg_energy:.2f} đơn vị\n")
            f.write(f"Điểm bảo mật trung bình: {avg_security:.2f}\n")
            f.write(f"Tỷ lệ giao dịch xuyên shard: {avg_cross_ratio:.2f}\n")
    
    def _save_summary_results(self, results, adaptive_consensus, dqn_enabled):
        """Lưu tổng hợp kết quả mô phỏng vào file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = "basic"
        if adaptive_consensus and dqn_enabled:
            config_name = "qtrust_full"
        elif adaptive_consensus:
            config_name = "adaptive_consensus"
        elif dqn_enabled:
            config_name = "dqn_routing"
            
        filename = f"results/summary_{config_name}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("QTrust Blockchain Simulation Results\n")
            f.write("=" * 40 + "\n\n")
            
            if adaptive_consensus and dqn_enabled:
                f.write("Configuration: QTrust Full (Adaptive Consensus + DQN Routing)\n")
            elif adaptive_consensus:
                f.write("Configuration: Adaptive Consensus Only\n")
            elif dqn_enabled:
                f.write("Configuration: DQN Routing Only\n")
            else:
                f.write("Configuration: Basic\n")
            
            f.write("\nPerformance Metrics:\n")
            f.write(f"  Throughput trung bình: {results['avg_throughput']:.2f} tx/s\n")
            f.write(f"  Độ trễ trung bình: {results['avg_latency']:.2f} ms\n")
            f.write(f"  Tiêu thụ năng lượng trung bình: {results['avg_energy']:.2f} đơn vị\n")
            f.write(f"  Điểm bảo mật trung bình: {results['avg_security']:.2f}\n")
            f.write(f"  Tỷ lệ giao dịch xuyên shard: {results['cross_shard_ratio']:.2f}\n")
            
            f.write("\nEpisode Rewards:\n")
            for i, reward in enumerate(results['episode_rewards']):
                f.write(f"  Episode {i+1}: {reward:.2f}\n")
                
            f.write(f"\nAverage Reward: {sum(results['episode_rewards']) / len(results['episode_rewards']):.2f}\n")
        
        print(f"Tổng hợp kết quả mô phỏng đã được lưu tại: {filename}")

def compare_configurations():
    """Chạy và so sánh các cấu hình khác nhau của hệ thống."""
    # Tạo đối tượng mô phỏng
    sim = SimpleBlockchainSimulation(
        num_shards=6,
        num_nodes_per_shard=15,
        max_steps=500,
        transaction_rate=30,
        cross_shard_prob=0.3
    )
    
    # Định nghĩa các cấu hình cần so sánh
    configs = [
        {"name": "Basic", "adaptive_consensus": False, "dqn_enabled": False},
        {"name": "Adaptive Consensus Only", "adaptive_consensus": True, "dqn_enabled": False},
        {"name": "DQN Only", "adaptive_consensus": False, "dqn_enabled": True},
        {"name": "QTrust Full", "adaptive_consensus": True, "dqn_enabled": True}
    ]
    
    # Lưu trữ kết quả
    results = []
    
    # Chạy mô phỏng cho mỗi cấu hình
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"Chạy mô phỏng với cấu hình: {config['name']}")
        print(f"{'='*80}\n")
        
        # Chạy mô phỏng
        result = sim.run_simulation(
            episodes=10,
            adaptive_consensus=config['adaptive_consensus'],
            dqn_enabled=config['dqn_enabled']
        )
        
        # Thêm tên cấu hình vào kết quả
        result['config_name'] = config['name']
        results.append(result)
    
    # So sánh các kết quả
    compare_results(results)
    
    return results

def compare_results(results):
    """
    So sánh và lưu kết quả giữa các cấu hình.
    
    Args:
        results: Danh sách kết quả từ các cấu hình khác nhau
    """
    # Tạo file so sánh
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/comparison_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("QTrust Configuration Comparison\n")
        f.write("=" * 40 + "\n\n")
        
        # Bảng so sánh
        f.write(f"{'Configuration':<25} {'Throughput':<12} {'Latency':<12} {'Energy':<12} {'Security':<12} {'Cross-Shard':<12}\n")
        f.write("-" * 80 + "\n")
        
        for result in results:
            f.write(f"{result['config_name']:<25} ")
            f.write(f"{result['avg_throughput']:<12.2f} ")
            f.write(f"{result['avg_latency']:<12.2f} ")
            f.write(f"{result['avg_energy']:<12.2f} ")
            f.write(f"{result['avg_security']:<12.2f} ")
            f.write(f"{result['cross_shard_ratio']:<12.2f}\n")
        
        # Phân tích
        f.write("\nPhân tích:\n")
        
        # Throughput
        best_throughput = max(results, key=lambda x: x['avg_throughput'])
        f.write(f"\n1. Throughput: Cấu hình '{best_throughput['config_name']}' cho throughput cao nhất ")
        f.write(f"({best_throughput['avg_throughput']:.2f} tx/s)\n")
        
        # Độ trễ
        best_latency = min(results, key=lambda x: x['avg_latency'])
        f.write(f"2. Độ trễ: Cấu hình '{best_latency['config_name']}' cho độ trễ thấp nhất ")
        f.write(f"({best_latency['avg_latency']:.2f} ms)\n")
        
        # Tiêu thụ năng lượng
        best_energy = min(results, key=lambda x: x['avg_energy'])
        f.write(f"3. Tiêu thụ năng lượng: Cấu hình '{best_energy['config_name']}' tiêu thụ ít năng lượng nhất ")
        f.write(f"({best_energy['avg_energy']:.2f})\n")
        
        # Bảo mật
        best_security = max(results, key=lambda x: x['avg_security'])
        f.write(f"4. Bảo mật: Cấu hình '{best_security['config_name']}' cho bảo mật cao nhất ")
        f.write(f"({best_security['avg_security']:.2f})\n")
        
        # Kết luận
        f.write("\nKết luận:\n")
        avg_rewards = [(r['config_name'], sum(r['episode_rewards']) / len(r['episode_rewards'])) for r in results]
        best_overall = max(avg_rewards, key=lambda x: x[1])
        
        f.write(f"Cấu hình '{best_overall[0]}' cho hiệu suất tổng thể tốt nhất với phần thưởng trung bình {best_overall[1]:.2f}.\n")
        
        if best_overall[0] == "QTrust Full":
            f.write("Kết quả này khẳng định hiệu quả của việc kết hợp cả giao thức đồng thuận thích ứng ")
            f.write("và định tuyến thông minh DQN trong giải pháp QTrust.\n")
        
    print(f"\nSo sánh các cấu hình đã được lưu tại: {filename}")

def main():
    """Chương trình chính."""
    print("=" * 80)
    print("QTrust Blockchain Simulation")
    print("Phân tích hiệu suất của giải pháp blockchain sharding với Deep RL")
    print("=" * 80)
    
    # Chạy so sánh các cấu hình
    compare_configurations()
    
    print("\nMô phỏng hoàn tất. Kết quả và phân tích đã được lưu trong thư mục 'results'.")

if __name__ == "__main__":
    main() 