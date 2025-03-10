"""
Module mô phỏng blockchain nâng cao tích hợp tất cả các cải tiến
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional
import json
import pandas as pd
from tqdm import tqdm

from dqn_blockchain_sim.blockchain.network import BlockchainNetwork
from dqn_blockchain_sim.agents.dqn_agent import ShardDQNAgent
from dqn_blockchain_sim.blockchain.mad_rapid import MADRAPIDProtocol
from dqn_blockchain_sim.tdcm.trust_manager import TrustManager
from dqn_blockchain_sim.blockchain.adaptive_consensus import AdaptiveCrossShardConsensus
from dqn_blockchain_sim.utils.real_data_builder import RealDataBuilder

class AdvancedSimulation:
    """
    Lớp mô phỏng nâng cao tích hợp tất cả các cải tiến
    """
    
    def __init__(
        self, 
        num_shards: int = 8,
        use_real_data: bool = False,
        use_dqn: bool = True,
        eth_api_key: str = None,
        data_dir: str = "data",
        log_dir: str = "logs"
    ):
        """
        Khởi tạo mô phỏng nâng cao
        
        Args:
            num_shards: Số lượng shard trong mạng
            use_real_data: Sử dụng dữ liệu thực từ Ethereum hay không
            use_dqn: Sử dụng DQN agent hay không
            eth_api_key: Khóa API Ethereum (chỉ cần khi use_real_data=True)
            data_dir: Thư mục lưu trữ dữ liệu
            log_dir: Thư mục lưu trữ nhật ký
        """
        self.num_shards = num_shards
        self.use_real_data = use_real_data
        self.use_dqn = use_dqn
        self.eth_api_key = eth_api_key
        self.data_dir = data_dir
        self.log_dir = log_dir
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Khởi tạo mạng blockchain
        blockchain_config = {"num_shards": num_shards, "num_nodes": num_shards * 10, 
                            "min_nodes_per_shard": 3, "block_time": 2.0,
                            "max_reshard_nodes": 5}
        self.network = BlockchainNetwork(config=blockchain_config)
        
        # Khởi tạo DQN agent
        self.dqn_agents = {}
        if use_dqn:
            for shard_id in range(num_shards):
                self.dqn_agents[shard_id] = ShardDQNAgent(
                    shard_id=shard_id,
                    state_dim=7,   # Điều chỉnh kích thước trạng thái phù hợp
                    action_dim=3   # Điều chỉnh kích thước hành động phù hợp
                )
        
        # Khởi tạo các module tích hợp
        # 1. HTDCM - Hierarchical Trust-based Data Center Mechanism
        self.htdcm = TrustManager(num_shards=num_shards, network=self.network)
        
        # 2. ACSC - Adaptive Cross-Shard Consensus
        self.acsc = AdaptiveCrossShardConsensus(network=self.network, trust_manager=self.htdcm)
        
        # 3. MAD-RAPID - Multi-Agent Dynamic RAPID Protocol
        self.mad_rapid = MADRAPIDProtocol(num_shards=num_shards, network=self.network)
        
        # 4. Real Data Integration
        self.real_data = None
        if use_real_data:
            self.real_data = RealDataBuilder(
                api_key=eth_api_key,
                data_dir=os.path.join(data_dir, "ethereum")
            )
        
        # Bộ đếm và thống kê
        self.current_step = 0
        self.total_transactions = 0
        self.successful_transactions = 0
        self.metrics_history = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'success_rate': [],
            'congestion': []
        }
        
        # Thông số mô phỏng
        self.default_tx_per_step = 20
        self.max_steps = 1000
        self.reset()
        
    def reset(self):
        """
        Đặt lại mô phỏng về trạng thái ban đầu
        """
        # Đặt lại mạng blockchain
        self.network.reset()
        
        # Đặt lại các module tích hợp
        if hasattr(self.htdcm, 'reset_statistics'):
            self.htdcm.reset_statistics()
        else:
            # Khởi tạo lại các thuộc tính thống kê
            self.htdcm.trust_scores = {i: 0.5 for i in range(self.num_shards)}
            self.htdcm.reputation_history = {}
        
        if hasattr(self.acsc, 'reset_statistics'):
            self.acsc.reset_statistics()
        else:
            # Khởi tạo lại các thuộc tính thống kê
            self.acsc.statistics = {
                'total_transactions': 0,
                'fast_consensus_count': 0,
                'standard_consensus_count': 0,
                'robust_consensus_count': 0,
                'success_count': 0
            }
        
        if hasattr(self.mad_rapid, 'reset_statistics'):
            self.mad_rapid.reset_statistics()
        else:
            # Khởi tạo lại các thuộc tính thống kê
            self.mad_rapid.performance_stats = {
                'total_transactions': 0,
                'optimized_transactions': 0,
                'latency_improvement': 0.0,
                'energy_saved': 0.0
            }
        
        if self.real_data and hasattr(self.real_data, 'reset_simulation_statistics'):
            self.real_data.reset_simulation_statistics()
            
        # Đặt lại DQN agents
        if self.use_dqn:
            for agent in self.dqn_agents.values():
                agent.reset()
                
        # Đặt lại bộ đếm và thống kê
        self.current_step = 0
        self.total_transactions = 0
        self.successful_transactions = 0
        for key in self.metrics_history:
            self.metrics_history[key] = []
    
    def _generate_transactions(self, num_tx: int) -> List[Any]:
        """
        Tạo các giao dịch cho bước mô phỏng hiện tại
        
        Args:
            num_tx: Số lượng giao dịch cần tạo
            
        Returns:
            Danh sách các giao dịch
        """
        transactions = []
        
        # Sử dụng dữ liệu thực nếu có
        if self.use_real_data and self.real_data and self.real_data.transaction_patterns:
            transactions = self.real_data.generate_batch(
                shard_count=self.num_shards,
                batch_size=num_tx
            )
        else:
            # Tạo dữ liệu mô phỏng
            for _ in range(num_tx):
                # Chọn shard nguồn và đích ngẫu nhiên
                source_shard = np.random.randint(0, self.num_shards)
                target_shard = np.random.randint(0, self.num_shards)
                
                # Đảm bảo target khác source cho giao dịch xuyên mảnh
                while target_shard == source_shard:
                    target_shard = np.random.randint(0, self.num_shards)
                    
                # Tạo giá trị giao dịch ngẫu nhiên
                value = np.random.exponential(100)  # Giá trị trung bình 100
                
                # Tạo giao dịch
                from dqn_blockchain_sim.blockchain.transaction import Transaction
                tx = Transaction(
                    source_shard=source_shard,
                    target_shard=target_shard,
                    value=value,
                    gas_limit=21000 + np.random.randint(0, 50000),
                    data={}
                )
                transactions.append(tx)
                
        self.total_transactions += len(transactions)
        return transactions
    
    def _update_network_state(self):
        """
        Cập nhật trạng thái mạng sau mỗi bước
        """
        # Cập nhật điểm tin cậy
        self.htdcm.update_trust_scores()
        
        # Cập nhật lịch sử đặc trưng của MAD-RAPID
        self.mad_rapid.update_feature_history()
        
        # Cập nhật các thống kê mạng
        for shard_id, shard in self.network.shards.items():
            # Cập nhật mức độ tắc nghẽn dựa trên kích thước hàng đợi
            queue_size = len(getattr(shard, 'transaction_queue', []))
            cross_queue_size = len(getattr(shard, 'cross_shard_queue', []))
            max_queue = 100  # Kích thước hàng đợi tối đa
            
            # Tính mức độ tắc nghẽn (0-1)
            shard.congestion_level = min(1.0, (queue_size + cross_queue_size * 2) / max_queue)
            
            # Cập nhật thông lượng
            processed_dict = getattr(shard, 'processed_transactions', {})
            processed = len(processed_dict) if isinstance(processed_dict, dict) else 0
            
            if hasattr(shard, 'prev_processed'):
                shard.throughput = processed - shard.prev_processed
            else:
                shard.throughput = 0
            shard.prev_processed = processed
            
            # Cập nhật độ trễ trung bình (giả định)
            base_latency = 50  # ms
            congestion_factor = 1 + shard.congestion_level * 5
            shard.avg_latency = base_latency * congestion_factor
    
    def _apply_dqn_actions(self):
        """
        Áp dụng hành động từ DQN agent cho mỗi shard
        """
        if not self.use_dqn:
            return
            
        for shard_id, agent in self.dqn_agents.items():
            shard = self.network.shards[shard_id]
            
            # Xây dựng trạng thái cho agent
            state = [
                shard.congestion_level,  # Mức độ tắc nghẽn
                len(getattr(shard, 'transaction_queue', [])) / 100,  # Kích thước hàng đợi thường (chuẩn hóa)
                len(getattr(shard, 'cross_shard_queue', [])) / 50,   # Kích thước hàng đợi xuyên mảnh (chuẩn hóa)
                getattr(shard, 'throughput', 0) / 50,                # Thông lượng (chuẩn hóa)
                self.htdcm.get_shard_trust_score(shard_id),          # Điểm tin cậy
                getattr(shard, 'avg_latency', 50) / 200,             # Độ trễ trung bình (chuẩn hóa)
                shard_id / self.num_shards                           # ID shard (chuẩn hóa)
            ]
            
            # Lấy hành động từ agent
            action = agent.select_action(state)
            
            # Áp dụng hành động
            if action == 0:
                # Tăng tốc xử lý giao dịch bình thường
                self._process_normal_transactions(shard, boost_factor=1.5)
            elif action == 1:
                # Tăng tốc xử lý giao dịch xuyên mảnh
                self._process_cross_shard_transactions(shard, boost_factor=1.5)
            else:
                # Cân bằng xử lý cả hai loại
                self._process_normal_transactions(shard, boost_factor=1.2)
                self._process_cross_shard_transactions(shard, boost_factor=1.2)
                
            # Tính phần thưởng
            reward = self._calculate_reward(shard)
            
            # Cập nhật agent
            next_state = state  # Đơn giản hóa, trong thực tế cần lấy trạng thái mới
            agent.update(state, action, reward, next_state, False)
    
    def _process_normal_transactions(self, shard, boost_factor=1.0):
        """
        Xử lý giao dịch thông thường trong shard
        
        Args:
            shard: Shard cần xử lý
            boost_factor: Hệ số tăng tốc
        """
        # Kiểm tra nếu có hàng đợi giao dịch
        if not hasattr(shard, 'transaction_queue') or not shard.transaction_queue:
            return
            
        # Tính số lượng giao dịch cần xử lý
        base_process_count = 5
        process_count = min(len(shard.transaction_queue), 
                          int(base_process_count * boost_factor))
        
        # Xử lý các giao dịch
        for _ in range(process_count):
            if not shard.transaction_queue:
                break
                
            # Lấy giao dịch đầu tiên
            tx = shard.transaction_queue.pop(0)
            
            # Thực hiện xác thực
            valid = True
            if hasattr(shard, 'validator') and hasattr(shard.validator, 'validate_transaction'):
                valid = shard.validator.validate_transaction(tx)
                
            if valid:
                # Giao dịch thành công
                if not hasattr(shard, 'processed_transactions'):
                    shard.processed_transactions = {}
                
                # Thêm giao dịch vào danh sách đã xử lý
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.processed_transactions[tx_id] = tx
                
                # Cập nhật thống kê
                if not hasattr(shard, 'confirmed_transactions'):
                    shard.confirmed_transactions = 0
                shard.confirmed_transactions += 1
                
                self.successful_transactions += 1
            else:
                # Giao dịch thất bại
                if not hasattr(shard, 'failed_transactions'):
                    shard.failed_transactions = {}
                
                # Thêm giao dịch vào danh sách thất bại
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.failed_transactions[tx_id] = tx
                
                # Cập nhật thống kê
                if not hasattr(shard, 'rejected_transactions'):
                    shard.rejected_transactions = 0
                shard.rejected_transactions += 1
    
    def _process_cross_shard_transactions(self, shard, boost_factor=1.0):
        """
        Xử lý giao dịch xuyên mảnh trong shard
        
        Args:
            shard: Shard cần xử lý
            boost_factor: Hệ số tăng tốc
        """
        # Kiểm tra nếu có hàng đợi giao dịch xuyên mảnh
        if not hasattr(shard, 'cross_shard_queue') or not shard.cross_shard_queue:
            return
            
        # Tính số lượng giao dịch cần xử lý
        base_process_count = 3
        process_count = min(len(shard.cross_shard_queue), 
                          int(base_process_count * boost_factor))
        
        # Nếu đã hook MAD-RAPID, nó sẽ xử lý tự động qua phương thức đã ghi đè
        # Nếu không, xử lý thủ công
        if not hasattr(shard, '_original_process_cross_shard'):
            # Xử lý thủ công
            for _ in range(process_count):
                if not shard.cross_shard_queue:
                    break
                    
                # Lấy giao dịch đầu tiên
                tx = shard.cross_shard_queue.pop(0)
                
                # Xử lý qua ACSC nếu có
                success = self.acsc.process_cross_shard_transaction(tx)
                
                if success:
                    # Giao dịch thành công
                    if not hasattr(shard, 'processed_transactions'):
                        shard.processed_transactions = {}
                    
                    # Thêm giao dịch vào danh sách đã xử lý
                    tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                    shard.processed_transactions[tx_id] = tx
                    
                    # Cập nhật thống kê
                    if not hasattr(shard, 'confirmed_count'):
                        shard.confirmed_count = 0
                    shard.confirmed_count += 1
                    
                    self.successful_transactions += 1
                else:
                    # Giao dịch thất bại
                    if not hasattr(shard, 'failed_transactions'):
                        shard.failed_transactions = {}
                    
                    # Thêm giao dịch vào danh sách thất bại
                    tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                    shard.failed_transactions[tx_id] = tx
                    
                    # Cập nhật thống kê
                    if not hasattr(shard, 'rejected_count'):
                        shard.rejected_count = 0
                    shard.rejected_count += 1
    
    def _calculate_reward(self, shard) -> float:
        """
        Tính phần thưởng cho DQN agent dựa trên hiệu suất của shard
        
        Args:
            shard: Shard cần tính phần thưởng
            
        Returns:
            Giá trị phần thưởng
        """
        # Thông số đánh giá
        throughput = getattr(shard, 'throughput', 0)
        congestion = getattr(shard, 'congestion_level', 0)
        latency = getattr(shard, 'avg_latency', 100)
        
        # Tính phần thưởng
        # - Thông lượng cao -> phần thưởng cao
        # - Tắc nghẽn thấp -> phần thưởng cao
        # - Độ trễ thấp -> phần thưởng cao
        reward = (throughput / 10) - congestion - (latency / 200)
        
        return reward
    
    def _update_metrics(self):
        """
        Cập nhật các metric mô phỏng
        """
        # Tính các metric trung bình trên tất cả các shard
        avg_throughput = 0
        avg_latency = 0
        avg_congestion = 0
        
        for shard in self.network.shards.values():
            avg_throughput += getattr(shard, 'throughput', 0)
            avg_latency += getattr(shard, 'avg_latency', 50)
            avg_congestion += getattr(shard, 'congestion_level', 0)
            
        avg_throughput /= self.num_shards
        avg_latency /= self.num_shards
        avg_congestion /= self.num_shards
        
        # Tính tỉ lệ thành công
        success_rate = 0
        if self.total_transactions > 0:
            success_rate = self.successful_transactions / self.total_transactions
            
        # Ước tính mức tiêu thụ năng lượng
        # Giả định từ các thông số của ACSC
        energy_consumption = self.acsc.stats.get("energy_usage", 0)
        
        # Lưu vào lịch sử
        self.metrics_history['throughput'].append(avg_throughput)
        self.metrics_history['latency'].append(avg_latency)
        self.metrics_history['energy_consumption'].append(energy_consumption)
        self.metrics_history['success_rate'].append(success_rate)
        self.metrics_history['congestion'].append(avg_congestion)
    
    def step(self, num_tx: int = None):
        """
        Thực hiện một bước mô phỏng
        
        Args:
            num_tx: Số lượng giao dịch trong bước này (tùy chọn)
            
        Returns:
            Thống kê của bước hiện tại
        """
        if num_tx is None:
            num_tx = self.default_tx_per_step
            
        # 1. Tạo giao dịch
        transactions = self._generate_transactions(num_tx)
        
        # 2. Phân phối giao dịch vào các shard
        for tx in transactions:
            source_shard = tx.source_shard
            target_shard = tx.target_shard
            
            if source_shard == target_shard:
                # Giao dịch trong cùng shard
                if not hasattr(self.network.shards[source_shard], 'transaction_queue'):
                    self.network.shards[source_shard].transaction_queue = []
                self.network.shards[source_shard].transaction_queue.append(tx)
            else:
                # Giao dịch xuyên mảnh
                if not hasattr(self.network.shards[source_shard], 'cross_shard_queue'):
                    self.network.shards[source_shard].cross_shard_queue = []
                self.network.shards[source_shard].cross_shard_queue.append(tx)
        
        # 3. Áp dụng các quyết định từ DQN agent
        self._apply_dqn_actions()
        
        # 4. Cập nhật trạng thái mạng
        self._update_network_state()
        
        # 5. Cập nhật các metric
        self._update_metrics()
        
        # 6. Tăng bước
        self.current_step += 1
        
        # 7. Trả về thống kê hiện tại
        return self._get_current_stats()
    
    def _get_current_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiện tại của mô phỏng
        
        Returns:
            Từ điển chứa các thống kê
        """
        step_stats = {
            'step': self.current_step,
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'success_rate': self.metrics_history['success_rate'][-1] if self.metrics_history['success_rate'] else 0,
            'avg_throughput': self.metrics_history['throughput'][-1] if self.metrics_history['throughput'] else 0,
            'avg_latency': self.metrics_history['latency'][-1] if self.metrics_history['latency'] else 0,
            'avg_congestion': self.metrics_history['congestion'][-1] if self.metrics_history['congestion'] else 0,
            'energy_consumption': self.metrics_history['energy_consumption'][-1] if self.metrics_history['energy_consumption'] else 0
        }
        
        # Thêm thống kê từ các module
        step_stats['mad_rapid'] = self.mad_rapid.get_statistics()
        step_stats['htdcm'] = self.htdcm.get_statistics()
        step_stats['acsc'] = self.acsc.get_statistics()
        
        if self.real_data:
            step_stats['real_data'] = self.real_data.get_statistics()
            
        return step_stats
    
    def run_simulation(self, num_steps: int = None, tx_per_step: int = None, visualize: bool = False, save_stats: bool = False):
        """
        Chạy mô phỏng với số bước và thông số đã cho
        
        Args:
            num_steps: Số bước mô phỏng
            tx_per_step: Số giao dịch mỗi bước
            visualize: Hiển thị đồ thị kết quả
            save_stats: Lưu thống kê vào file
            
        Returns:
            Thống kê tổng hợp của mô phỏng
        """
        if num_steps is None:
            num_steps = self.max_steps
            
        if tx_per_step is not None:
            self.default_tx_per_step = tx_per_step
            
        # Đặt lại mô phỏng
        self.reset()
        
        # Tải dữ liệu thực nếu cần
        if self.use_real_data and self.real_data:
            data_file = os.path.join(self.data_dir, "ethereum", "eth_transactions.csv")
            self.real_data.load_data(file_path=data_file)
        
        # Chạy mô phỏng
        all_stats = []
        for _ in tqdm(range(num_steps), desc="Mô phỏng"):
            step_stats = self.step()
            all_stats.append(step_stats)
            
        # Hiển thị kết quả nếu cần
        if visualize:
            self.visualize_results()
            
        # Lưu thống kê nếu cần
        if save_stats:
            self.save_statistics()
            
        # Trả về thống kê tổng hợp
        return self.get_summary_statistics()
    
    def visualize_results(self):
        """
        Hiển thị kết quả mô phỏng dưới dạng đồ thị
        """
        if not self.metrics_history:
            print("Khong co du lieu de hien thi!")
            return
            
        # Tạo các biểu đồ
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Thông lượng
        axs[0, 0].plot([m['avg_throughput'] for m in self.metrics_history])
        axs[0, 0].set_title('Throughput')
        axs[0, 0].set_xlabel('Step')
        axs[0, 0].set_ylabel('Transactions/step')
        
        # 2. Độ trễ
        axs[0, 1].plot([m['avg_latency'] for m in self.metrics_history])
        axs[0, 1].set_title('Average Latency')
        axs[0, 1].set_xlabel('Step')
        axs[0, 1].set_ylabel('Time (ms)')
        
        # 3. Tỉ lệ thành công
        axs[1, 0].plot([m['success_rate'] for m in self.metrics_history])
        axs[1, 0].set_title('Success Rate')
        axs[1, 0].set_xlabel('Step')
        axs[1, 0].set_ylabel('Rate')
        
        # 4. Mức độ tắc nghẽn
        axs[1, 1].plot([m['avg_congestion'] for m in self.metrics_history])
        axs[1, 1].set_title('Average Congestion')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('Level')
        
        plt.tight_layout()
        
        # Biểu đồ so sánh loại đồng thuận
        if hasattr(self, 'acsc') and hasattr(self.acsc, 'get_statistics'):
            stats = self.acsc.get_statistics()
            if 'strategy_usage' in stats:
                fig, ax = plt.subplots(figsize=(10, 6))
                strategies = list(stats['strategy_usage'].keys())
                usage = list(stats['strategy_usage'].values())
                
                ax.bar(strategies, usage)
                ax.set_title('Consensus Type Distribution')
                ax.set_xlabel('Strategy')
                ax.set_ylabel('Usage Count')
                
        plt.show()
    
    def save_statistics(self):
        """
        Lưu thống kê mô phỏng vào file
        """
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Lưu thống kê tổng hợp
        summary_stats = self.get_summary_statistics()
        timestamp = int(time.time())
        summary_file = os.path.join(self.log_dir, f"summary_stats_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2, ensure_ascii=False)
        print(f"Da luu thong ke tong hop vao {summary_file}")
        
        # Lưu lịch sử metrics
        metrics_file = os.path.join(self.log_dir, f"metrics_history_{timestamp}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
        print(f"Da luu lich su metrics vao {metrics_file}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp của mô phỏng
        
        Returns:
            Từ điển chứa các thống kê tổng hợp
        """
        # Tính giá trị trung bình của các metric
        avg_metrics = {}
        for key, values in self.metrics_history.items():
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
                avg_metrics[f"max_{key}"] = max(values)
                avg_metrics[f"min_{key}"] = min(values)
        
        # Tổng hợp thống kê
        summary = {
            'simulation_config': {
                'num_shards': self.num_shards,
                'use_real_data': self.use_real_data,
                'use_dqn': self.use_dqn,
                'steps': self.current_step,
                'transactions_per_step': self.default_tx_per_step
            },
            'transaction_stats': {
                'total_transactions': self.total_transactions,
                'successful_transactions': self.successful_transactions,
                'success_rate': self.successful_transactions / max(1, self.total_transactions)
            },
            'performance_metrics': avg_metrics,
            'module_stats': {
                'mad_rapid': self.mad_rapid.get_statistics(),
                'htdcm': self.htdcm.get_statistics(),
                'acsc': self.acsc.get_statistics()
            }
        }
        
        if self.real_data:
            summary['real_data'] = self.real_data.get_statistics()
            
        return summary

    def integrate_mad_rapid(self):
        """Tích hợp giao thức MAD-RAPID"""
        self.mad_rapid = MADRAPIDProtocol(
            num_shards=self.num_shards,
            network=self.network
        )
        
    def integrate_htdcm(self):
        """Tích hợp cơ chế HTDCM"""
        self.trust_manager = TrustManager(
            num_shards=self.num_shards,
            network=self.network
        )
        
    def integrate_acsc(self):
        """Tích hợp cơ chế ACSC"""
        self.consensus = AdaptiveCrossShardConsensus(
            network=self.network,
            trust_manager=self.trust_manager
        )
        
    def integrate_real_data(self):
        """Tích hợp dữ liệu thực từ Ethereum"""
        if not self.eth_api_key:
            raise ValueError("Cần cung cấp eth_api_key để sử dụng dữ liệu thực")
        
        self.data_builder = RealDataBuilder(
            api_key=self.eth_api_key,
            data_dir=self.data_dir
        )
        self.transactions = self.data_builder.build_transactions()


def run_advanced_simulation(args):
    """
    Hàm main để chạy mô phỏng nâng cao
    
    Args:
        args: Tham số dòng lệnh
    """
    # Khởi tạo mô phỏng
    simulation = AdvancedSimulation(
        num_shards=args.num_shards,
        use_real_data=args.use_real_data,
        use_dqn=args.use_dqn,
        eth_api_key=args.eth_api_key,
        data_dir=args.data_dir,
        log_dir=args.log_dir
    )
    
    # Chạy mô phỏng
    simulation.run_simulation(
        num_steps=args.steps,
        tx_per_step=args.tx_per_step,
        visualize=args.visualize,
        save_stats=args.save_stats
    )
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Chạy mô phỏng blockchain nâng cao')
    parser.add_argument('--num_shards', type=int, default=8, help='Số lượng shard')
    parser.add_argument('--steps', type=int, default=1000, help='Số bước mô phỏng')
    parser.add_argument('--tx_per_step', type=int, default=20, help='Số giao dịch mỗi bước')
    parser.add_argument('--use_real_data', action='store_true', help='Sử dụng dữ liệu Ethereum thực')
    parser.add_argument('--use_dqn', action='store_true', help='Sử dụng DQN agent')
    parser.add_argument('--eth_api_key', type=str, help='Khóa API Ethereum (nếu sử dụng dữ liệu thực)')
    parser.add_argument('--data_dir', type=str, default='data', help='Thư mục lưu dữ liệu')
    parser.add_argument('--log_dir', type=str, default='logs', help='Thư mục lưu nhật ký')
    parser.add_argument('--visualize', action='store_true', help='Hiển thị đồ thị kết quả')
    parser.add_argument('--save_stats', action='store_true', help='Lưu thống kê vào file')
    
    args = parser.parse_args()
    
    run_advanced_simulation(args) 