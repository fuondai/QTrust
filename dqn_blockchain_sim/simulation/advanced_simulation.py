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
import random

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
                    state_size=7,   # Điều chỉnh kích thước trạng thái phù hợp
                    action_size=3   # Điều chỉnh kích thước hành động phù hợp
                )
        
        # Khởi tạo các module tích hợp
        # 1. HTDCM - Hierarchical Trust-based Data Center Mechanism
        self.trust_manager = TrustManager(num_shards=num_shards, network=self.network)
        
        # 2. ACSC - Adaptive Cross-Shard Consensus
        # Initialize HTDCM (Hierarchical Trust-based Data Center Mechanism)
        self.htdcm = TrustManager(network=self.network, num_shards=num_shards)
        
        # Initialize the adaptive cross-shard consensus mechanism
        # Fix: Remove the 'network' parameter if it's not expected
        self.acsc = AdaptiveCrossShardConsensus(trust_manager=self.trust_manager)
        
        # 3. MAD-RAPID - Multi-Agent Dynamic RAPID Protocol
        self.mad_rapid = MADRAPIDProtocol(
            input_size=8,
            embedding_dim=64,
            hidden_size=128,
            lookback_window=10,
            prediction_horizon=5
        )
        
        # Kết nối MAD-RAPID với mạng
        self._hook_mad_rapid()
        
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
        self.stats = {
            "cross_shard_transactions": 0,
            "failed_transactions": 0,
            "mad_rapid_optimized": 0,
            "htdcm_transactions": 0,
            "acsc_energy_saved": 0.0
        }
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
        
        # Khởi tạo hoặc đặt lại transaction_data
        self.transaction_data = {}
        
        # Đặt lại các module tích hợp
        if hasattr(self.trust_manager, 'reset_statistics'):
            self.trust_manager.reset_statistics()
        else:
            # Khởi tạo lại các thuộc tính thống kê
            self.trust_manager.trust_scores = {i: 0.5 for i in range(self.num_shards)}
            self.trust_manager.reputation_history = {}
        
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
                # Chọn shard nguồn ngẫu nhiên
                source_shard = np.random.randint(0, self.num_shards)
                
                # Xác định xem đây có phải là giao dịch xuyên mảnh hay không (80% khả năng)
                is_cross_shard = np.random.random() < 0.8
                
                if is_cross_shard:
                    # Đảm bảo target khác source cho giao dịch xuyên mảnh
                    target_shard = np.random.randint(0, self.num_shards)
                    while target_shard == source_shard:
                        target_shard = np.random.randint(0, self.num_shards)
                else:
                    # Giao dịch nội bộ có cùng shard nguồn và đích
                    target_shard = source_shard
                    
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
                
                # Đảm bảo thuộc tính is_cross_shard được thiết lập đúng
                tx.is_cross_shard = (source_shard != target_shard)
                
                transactions.append(tx)
                
        self.total_transactions += len(transactions)
        return transactions
    
    def _update_network_state(self):
        """
        Cập nhật trạng thái mạng sau mỗi bước
        """
        # Cập nhật điểm tin cậy
        self.trust_manager.update_trust_scores()
        
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
                self.trust_manager.get_shard_trust_score(shard_id),          # Điểm tin cậy
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
        """Xử lý các giao dịch thường trong shard"""
        if not hasattr(shard, 'transaction_queue'):
            shard.transaction_queue = []
        
        # Thiết lập các thuộc tính nếu chưa tồn tại
        for attr in ['processed_transactions', 'failed_transactions', 'confirmed_transactions', 
                    'rejected_transactions', 'performance']:
            if not hasattr(shard, attr):
                if attr == 'performance':
                    setattr(shard, attr, 0.8)  # Giá trị mặc định cho performance
                else:
                    setattr(shard, attr, {})
        
        if not hasattr(shard, 'successful_tx_count'):
            shard.successful_tx_count = 0
        
        # Tính toán số lượng giao dịch có thể xử lý dựa trên hiệu suất shard
        processing_capacity = int(shard.performance * boost_factor)
        processing_capacity = max(1, processing_capacity)  # Đảm bảo ít nhất 1 giao dịch được xử lý
        
        if len(shard.transaction_queue) == 0:
            return
        
        # Lấy danh sách giao dịch để xử lý
        transactions_to_process = shard.transaction_queue[:processing_capacity]
        
        # Xóa các giao dịch đã xử lý khỏi hàng đợi
        shard.transaction_queue = shard.transaction_queue[processing_capacity:]
        
        # Tỷ lệ thành công dựa trên hiệu suất shard
        base_success_rate = 0.7 + (shard.performance * 0.3)
        
        # Xử lý từng giao dịch
        for tx in transactions_to_process:
            # Xác định xem giao dịch có thành công hay không
            is_successful = random.random() < base_success_rate
            gas_used = getattr(tx, 'gas_limit', 21000)
            fee = getattr(tx, 'gas_price', 1.0) * gas_used
            
            if is_successful:
                # Cập nhật số liệu cho giao dịch thành công
                shard.successful_tx_count += 1
                self.successful_transactions += 1
                
                # Thêm vào từ điển giao dịch đã xử lý
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.processed_transactions[tx_id] = tx
                shard.confirmed_transactions[tx_id] = tx
                
                # Cập nhật các biến thống kê
                if hasattr(shard, 'throughput'):
                    shard.throughput += 1
                else:
                    shard.throughput = 1
                    
                if hasattr(shard, 'total_gas_used'):
                    shard.total_gas_used += gas_used
                else:
                    shard.total_gas_used = gas_used
                    
                if hasattr(shard, 'total_fees'):
                    shard.total_fees += fee
                else:
                    shard.total_fees = fee
                    
                # Thêm giao dịch vào transaction_data
                tx.status = "processed"
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                self.transaction_data[tx_id] = {
                    'status': 'processed',
                    'is_cross_shard': False,
                    'source_shard': tx.source_shard,
                    'target_shard': tx.target_shard,
                    'timestamp': time.time(),
                    'size': getattr(tx, 'size', 1024),
                    'value': getattr(tx, 'value', 0.0),
                    'gas_limit': getattr(tx, 'gas_limit', 21000),
                    'gas_price': getattr(tx, 'gas_price', 1.0)
                }
            else:
                # Xử lý trường hợp giao dịch thất bại
                tx.status = "failed"
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.failed_transactions[tx_id] = tx
                if hasattr(shard, 'rejected_transactions'):
                    shard.rejected_transactions[tx_id] = tx
                    
                # Thêm giao dịch vào transaction_data với trạng thái failed
                self.transaction_data[tx_id] = {
                    'status': 'failed',
                    'is_cross_shard': False,
                    'source_shard': tx.source_shard,
                    'target_shard': tx.target_shard,
                    'timestamp': time.time(),
                    'size': getattr(tx, 'size', 1024),
                    'value': getattr(tx, 'value', 0.0),
                    'gas_limit': getattr(tx, 'gas_limit', 21000),
                    'gas_price': getattr(tx, 'gas_price', 1.0)
                }
    
    def _process_cross_shard_transactions(self, shard, boost_factor=1.0):
        """
        Xử lý các giao dịch xuyên shard trong hàng đợi
        
        Args:
            shard: Shard cần xử lý
            boost_factor: Hệ số tăng cường hiệu suất
        """
        # Đảm bảo các thuộc tính cần thiết tồn tại
        if not hasattr(shard, 'cross_shard_queue'):
            shard.cross_shard_queue = []
            
        if not hasattr(shard, 'cross_shard_tx_count'):
            shard.cross_shard_tx_count = 0
            
        # Thiết lập các thuộc tính khác nếu chưa tồn tại
        for attr in ['processed_transactions', 'failed_transactions', 'confirmed_transactions', 'rejected_transactions']:
            if not hasattr(shard, attr):
                setattr(shard, attr, {})
                
        if not hasattr(shard, 'successful_cs_tx_count'):
            shard.successful_cs_tx_count = 0
        
        # Đảm bảo thuộc tính performance tồn tại
        if not hasattr(shard, 'performance'):
            shard.performance = 1.0
        
        if len(shard.cross_shard_queue) == 0:
            return
        
        # Tính toán công suất xử lý dựa trên hiệu suất shard
        base_process_count = 5  # Tăng từ 3 lên 5 để xử lý nhiều giao dịch hơn
        processing_capacity = int(base_process_count * shard.performance * boost_factor)
        
        # Lấy danh sách giao dịch để xử lý
        transactions_to_process = shard.cross_shard_queue[:processing_capacity]
        
        # Xóa các giao dịch khỏi hàng đợi
        shard.cross_shard_queue = shard.cross_shard_queue[processing_capacity:]
        
        # Kiểm tra xem MAD-RAPID đã được khởi tạo chưa
        if not hasattr(self, 'mad_rapid') or self.mad_rapid is None:
            print("MAD-RAPID chưa được khởi tạo, đang khởi tạo...")
            self.integrate_mad_rapid()
        
        # Kiểm tra xem ACSC đã được khởi tạo chưa
        if not hasattr(self, 'acsc') or self.acsc is None:
            print("ACSC chưa được khởi tạo, đang khởi tạo...")
            self.integrate_acsc()
        
        # Đảm bảo HTDCM đã được khởi tạo
        if not hasattr(self, 'htdcm') or self.htdcm is None:
            print("HTDCM chưa được khởi tạo, đang khởi tạo...")
            self.integrate_htdcm()
        
        # Cập nhật điểm tin cậy trước khi xử lý giao dịch
        try:
            self.htdcm.update_trust_scores()
            print(f"Đã cập nhật điểm tin cậy cho {len(self.htdcm.trust_scores)} node")
        except Exception as e:
            print(f"Lỗi khi cập nhật điểm tin cậy: {e}")
        
        print(f"Đang xử lý {len(transactions_to_process)} giao dịch xuyên shard cho shard {shard.shard_id}")
        
        # Xử lý từng giao dịch
        mad_rapid_success_count = 0
        acsc_success_count = 0
        
        for tx in transactions_to_process:
            # Tăng số lượng giao dịch xuyên shard
            shard.cross_shard_tx_count += 1
            
            # Đảm bảo giao dịch có các thuộc tính cần thiết
            if not hasattr(tx, 'source_shard'):
                tx.source_shard = shard.shard_id
            
            if not hasattr(tx, 'is_cross_shard'):
                tx.is_cross_shard = True
            
            # Quyết định phương pháp xử lý dựa trên giá trị giao dịch và độ tin cậy
            # Sử dụng DQN Agent để quyết định nếu có thể
            use_acsc = False
            
            # Sử dụng DQN nếu đã được kích hoạt
            if hasattr(self, 'use_dqn') and self.use_dqn and shard.shard_id in self.dqn_agents:
                # Lấy đặc trưng trạng thái
                tx_value = getattr(tx, 'value', 0)
                cross_shard_level = shard.cross_shard_tx_count / max(1, len(transactions_to_process)) 
                congestion_level = len(shard.cross_shard_queue) / max(10, processing_capacity)
                
                # Tạo vector trạng thái
                state = np.array([tx_value, cross_shard_level, congestion_level])
                
                # Dự đoán hành động (0: MAD-RAPID, 1: ACSC)
                action = self.dqn_agents[shard.shard_id].predict(state)
                use_acsc = (action == 1)
                
                # Cập nhật trạng thái DQN
                if hasattr(self.dqn_agents[shard.shard_id], 'current_state'):
                    self.dqn_agents[shard.shard_id].current_state = state
            else:
                # Nếu không có DQN, quyết định dựa trên heuristic
                tx_value = getattr(tx, 'value', 0)
                target_shard = getattr(tx, 'target_shard', 0)
                
                # Kiểm tra độ tin cậy giữa các shard
                trust_level = self.htdcm.get_trust_between_shards(shard.shard_id, target_shard) if hasattr(self.htdcm, 'get_trust_between_shards') else 0.5
                
                # Sử dụng ACSC cho giao dịch có giá trị cao và độ tin cậy thấp
                use_acsc = (tx_value > 100 and trust_level < 0.7)
            
            is_successful = False
            
            if use_acsc:
                print(f"Sử dụng ACSC để xử lý giao dịch {getattr(tx, 'transaction_id', str(id(tx)))}")
                try:
                    # Xử lý bằng ACSC
                    is_successful, details = self.acsc.process_cross_shard_transaction(
                        transaction=tx,
                        source_shard_id=shard.shard_id,
                        target_shard_id=getattr(tx, 'target_shard', 0),
                        network=self.network
                    )
                    
                    # Cập nhật thống kê
                    if is_successful:
                        acsc_success_count += 1
                except Exception as e:
                    print(f"Lỗi khi xử lý giao dịch bằng ACSC: {e}")
                    # Thử lại với MAD-RAPID nếu ACSC thất bại
                    is_successful = self.mad_rapid.process_cross_shard_transaction(tx)
                    if is_successful:
                        mad_rapid_success_count += 1
            else:
                # Sử dụng MAD-RAPID
                print(f"Sử dụng MAD-RAPID để xử lý giao dịch {getattr(tx, 'transaction_id', str(id(tx)))}")
                is_successful = self.mad_rapid.process_cross_shard_transaction(tx)
                if is_successful:
                    mad_rapid_success_count += 1
            
            # Tính toán gas và phí
            gas_used = getattr(tx, 'gas_limit', 21000) * 1.5  # Giao dịch xuyên shard tiêu thụ gas cao hơn
            fee = getattr(tx, 'gas_price', 1.0) * gas_used
            
            if is_successful:
                # Cập nhật số liệu cho giao dịch thành công
                shard.successful_cs_tx_count += 1
                self.successful_transactions += 1
                
                # Thêm vào từ điển giao dịch đã xử lý
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.processed_transactions[tx_id] = tx
                
                # Cập nhật thưởng cho node
                validator_reward = fee * 0.7  # 70% phí cho validator
                
                # Thưởng cho DQN nếu đã sử dụng và dự đoán đúng
                if hasattr(self, 'use_dqn') and self.use_dqn and shard.shard_id in self.dqn_agents:
                    if (use_acsc and is_successful) or (not use_acsc and is_successful):
                        # DQN đã dự đoán đúng, thưởng cho nó
                        if hasattr(self.dqn_agents[shard.shard_id], 'reward'):
                            self.dqn_agents[shard.shard_id].reward(1.0)  # Thưởng tích cực
            else:
                # Giao dịch thất bại
                tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                shard.failed_transactions[tx_id] = tx
                
                # Phạt cho DQN nếu đã sử dụng và dự đoán sai
                if hasattr(self, 'use_dqn') and self.use_dqn and shard.shard_id in self.dqn_agents:
                    if hasattr(self.dqn_agents[shard.shard_id], 'reward'):
                        self.dqn_agents[shard.shard_id].reward(-0.5)  # Phạt nhẹ
        
        # Huấn luyện DQN sau mỗi batch giao dịch
        if hasattr(self, 'use_dqn') and self.use_dqn:
            for agent_id, agent in self.dqn_agents.items():
                if hasattr(agent, 'train'):
                    agent.train()
        
        print(f"MAD-RAPID đã xử lý thành công {mad_rapid_success_count}/{len(transactions_to_process)} giao dịch")
        print(f"ACSC đã xử lý thành công {acsc_success_count}/{len(transactions_to_process)} giao dịch")
        print(f"Tổng số giao dịch thành công: {mad_rapid_success_count + acsc_success_count}/{len(transactions_to_process)}")
    
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
        """Cập nhật các metrics của mô phỏng"""
        # Thiết lập các metrics mặc định nếu chưa tồn tại
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {
                'throughput': [],
                'latency': [],
                'energy_consumption': [],
                'success_rate': [],
                'congestion': []
            }
        
        # Tính toán các metrics từ dữ liệu trong mỗi shard
        total_throughput = 0
        total_latency = 0
        total_energy = 0
        total_congestion = 0
        active_shards = 0
        
        for shard_id, shard in self.network.shards.items():
            # Đảm bảo các thuộc tính cơ bản tồn tại
            if not hasattr(shard, 'performance'):
                shard.performance = 0.8  # Giá trị mặc định

            if not hasattr(shard, 'transaction_queue'):
                shard.transaction_queue = []
        
            # Cập nhật throughput - số giao dịch được xử lý mỗi giây
            shard_throughput = getattr(shard, 'throughput', 0)
            total_throughput += shard_throughput
            
            # Tính toán độ trễ (latency) dựa trên hiệu suất shard và số lượng giao dịch xử lý
            # Độ trễ tăng khi hiệu suất giảm và khi xử lý nhiều giao dịch
            base_latency = 10  # ms
            performance_factor = 1.0 / max(0.1, shard.performance)
            queue_size = len(getattr(shard, 'transaction_queue', [])) + len(getattr(shard, 'cross_shard_queue', []))
            queue_factor = 1.0 + (queue_size / 100)  # Độ trễ tăng khi hàng đợi dài
            
            shard_latency = base_latency * performance_factor * queue_factor
            shard_latency = max(5, min(500, shard_latency))  # Giới hạn trong khoảng [5, 500] ms
            
            if hasattr(shard, 'latency'):
                shard.latency = shard_latency
            else:
                shard.latency = shard_latency
            
            total_latency += shard_latency
            
            # Tính toán mức tiêu thụ năng lượng dựa trên số giao dịch xử lý và phức tạp của chúng
            # Năng lượng tiêu thụ tỉ lệ với số lượng giao dịch và độ phức tạp của chúng
            base_energy_per_tx = 5  # Đơn vị năng lượng cơ bản cho mỗi giao dịch
            cs_energy_factor = 2.0  # Giao dịch xuyên shard tiêu thụ năng lượng gấp đôi
            
            # Số lượng giao dịch đã xử lý
            normal_tx_count = getattr(shard, 'successful_tx_count', 0)
            cs_tx_count = getattr(shard, 'successful_cs_tx_count', 0)
            
            # Tính toán năng lượng
            normal_energy = normal_tx_count * base_energy_per_tx
            cs_energy = cs_tx_count * base_energy_per_tx * cs_energy_factor
            
            # Tổng năng lượng cho shard
            shard_energy = normal_energy + cs_energy
            
            # Thêm chi phí validation/mining
            if hasattr(shard, 'validator') and shard_throughput > 0:
                validation_energy = shard_throughput * 10  # Chi phí năng lượng cho validation
                shard_energy += validation_energy
            
            if hasattr(shard, 'total_gas_used'):
                gas_energy = shard.total_gas_used * 0.01  # Chuyển đổi gas thành năng lượng
                shard_energy += gas_energy
            
            # Lưu năng lượng vào shard
            if hasattr(shard, 'energy_consumption'):
                shard.energy_consumption = shard_energy
            else:
                shard.energy_consumption = shard_energy
            
            total_energy += shard_energy
            
            # Tính toán congestion (tắc nghẽn) dựa trên độ dài hàng đợi và throughput
            if not hasattr(shard, 'congestion_level'):
                shard.congestion_level = 0.0
            
            if shard_throughput > 0:
                queue_congestion = queue_size / (shard_throughput * 5)  # Tỷ lệ hàng đợi so với khả năng xử lý
                queue_congestion = min(1.0, queue_congestion)  # Giới hạn tối đa là 100%
            else:
                queue_congestion = min(1.0, queue_size / 20)  # Nếu không có throughput, dựa vào kích thước hàng đợi
            
            # Thêm yếu tố ngẫu nhiên để mô phỏng những biến động mạng
            random_factor = 1.0 + (random.random() * 0.2 - 0.1)  # Biến động ±10%
            shard_congestion = queue_congestion * random_factor
            shard_congestion = max(0.0, min(1.0, shard_congestion))  # Giới hạn trong khoảng [0, 1]
            
            shard.congestion_level = shard_congestion
            
            total_congestion += shard_congestion
            active_shards += 1
        
        # Tính giá trị trung bình
        avg_throughput = total_throughput / max(1, len(self.network.shards))
        avg_latency = total_latency / max(1, active_shards)
        avg_energy = total_energy
        avg_congestion = total_congestion / max(1, active_shards)
        
        # Tính tỷ lệ thành công
        if self.total_transactions > 0:
            success_rate = self.successful_transactions / self.total_transactions
        else:
            success_rate = 0.0
        
        # Thêm vào lịch sử metrics
        self.metrics_history['throughput'].append(avg_throughput)
        self.metrics_history['latency'].append(avg_latency)
        self.metrics_history['energy_consumption'].append(avg_energy)
        self.metrics_history['success_rate'].append(success_rate)
        self.metrics_history['congestion'].append(avg_congestion)
        
        # Cập nhật các biến trạng thái
        self.current_throughput = avg_throughput
        self.current_latency = avg_latency
        self.current_energy = avg_energy
        self.current_success_rate = success_rate
        self.current_congestion = avg_congestion
    
    def run_step(self, num_transactions: int = None) -> Dict[str, Any]:
        """
        Chạy một bước mô phỏng
        
        Args:
            num_transactions: Số lượng giao dịch trong bước này
            
        Returns:
            Kết quả bước mô phỏng
        """
        # Tính số lượng giao dịch
        if num_transactions is None:
            num_transactions = self.default_tx_per_step
        
        # Cập nhật thông tin mạng
        self._update_network_state()
        
        # Đảm bảo MAD-RAPID được kết nối đúng cách
        self._hook_mad_rapid()
        
        # Tạo giao dịch
        transactions = self._generate_transactions(num_transactions)
        
        # Đếm số lượng giao dịch cross-shard
        cross_shard_count = sum(1 for tx in transactions if tx.source_shard != tx.target_shard)
        print(f"Tạo {len(transactions)} giao dịch, trong đó có {cross_shard_count} giao dịch cross-shard")
        
        # Kiểm tra MAD-RAPID
        if hasattr(self, 'mad_rapid') and self.mad_rapid is not None:
            print(f"MAD-RAPID đã được khởi tạo: {self.mad_rapid}")
            print(f"MAD-RAPID có tham chiếu đến simulation: {hasattr(self.mad_rapid, 'simulation')}")
            print(f"MAD-RAPID có tham chiếu đến network: {hasattr(self.mad_rapid, 'network')}")
        else:
            print("MAD-RAPID chưa được khởi tạo!")
            # Khởi tạo lại MAD-RAPID nếu chưa được khởi tạo
            self.integrate_mad_rapid()
        
        # Phân phối giao dịch
        mad_rapid_processed = 0
        for tx in transactions:
            if tx.source_shard == tx.target_shard:
                # Giao dịch nội bộ
                if tx.source_shard in self.network.shards:
                    shard = self.network.shards[tx.source_shard]
                    if not hasattr(shard, 'transaction_queue'):
                        shard.transaction_queue = []
                    shard.transaction_queue.append(tx)
            else:
                # Giao dịch xuyên mảnh
                # Sử dụng MAD-RAPID nếu có
                if hasattr(self, 'mad_rapid') and self.mad_rapid is not None:
                    print(f"Xử lý giao dịch cross-shard từ shard {tx.source_shard} đến shard {tx.target_shard}")
                    optimized = self.mad_rapid.process_cross_shard_transaction(tx)
                    print(f"Kết quả tối ưu hóa: {optimized}")
                    if optimized:
                        # Đặt trạng thái giao dịch là đã xử lý
                        tx.status = "processed"
                        
                        # Cập nhật số lượng giao dịch thành công
                        self.successful_transactions += 1
                        mad_rapid_processed += 1
                        
                        # Cập nhật transaction_data
                        tx_id = getattr(tx, 'transaction_id', str(id(tx)))
                        self.transaction_data[tx_id] = {
                            'status': 'processed',
                            'is_cross_shard': True,
                            'source_shard': tx.source_shard,
                            'target_shard': tx.target_shard,
                            'timestamp': time.time(),
                            'size': getattr(tx, 'size', 1024),
                            'value': getattr(tx, 'value', 0.0),
                            'gas_limit': getattr(tx, 'gas_limit', 21000),
                            'gas_price': getattr(tx, 'gas_price', 1.0)
                        }
                        
                        self.stats["mad_rapid_optimized"] += 1
                        continue
                
                # Nếu không sử dụng MAD-RAPID hoặc MAD-RAPID không tối ưu được,
                # thêm vào hàng đợi xuyên mảnh
                if tx.source_shard in self.network.shards:
                    shard = self.network.shards[tx.source_shard]
                    if not hasattr(shard, 'cross_shard_queue'):
                        shard.cross_shard_queue = []
                    shard.cross_shard_queue.append(tx)
        
        print(f"MAD-RAPID đã xử lý {mad_rapid_processed}/{cross_shard_count} giao dịch cross-shard")
        
        # Sử dụng DQN để quản lý xử lý giao dịch
        if self.use_dqn:
            self._apply_dqn_actions()
        else:
            # Xử lý các giao dịch nếu không dùng DQN
            for shard_id, shard in self.network.shards.items():
                # Xử lý giao dịch thông thường
                boost_factor = 1.0
                if hasattr(shard, 'performance_boost'):
                    boost_factor = shard.performance_boost
                self._process_regular_transactions(shard, boost_factor)
                
                # Xử lý giao dịch xuyên mảnh
                self._process_cross_shard_transactions(shard, boost_factor)
        
        # Cập nhật thông tin mạng
        self._update_network_state()
        
        # Tính toán và ghi lại các chỉ số hiệu suất
        self._update_metrics()
        
        # Tăng bước mô phỏng
        self.current_step += 1
        
        # Trả về thống kê bước hiện tại
        return self.get_summary_statistics()
    
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
        step_stats['trust_manager'] = self.trust_manager.get_statistics()
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
            step_stats = self.run_step()
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
        if not self.metrics_history['throughput']:
            print("Không có dữ liệu để hiển thị!")
            return
            
        # Tạo các biểu đồ
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Thông lượng
        axs[0, 0].plot(self.metrics_history['throughput'])
        axs[0, 0].set_title('Thông lượng')
        axs[0, 0].set_xlabel('Bước')
        axs[0, 0].set_ylabel('Giao dịch/bước')
        
        # 2. Độ trễ
        axs[0, 1].plot(self.metrics_history['latency'])
        axs[0, 1].set_title('Độ trễ trung bình')
        axs[0, 1].set_xlabel('Bước')
        axs[0, 1].set_ylabel('Thời gian (ms)')
        
        # 3. Tỉ lệ thành công
        axs[1, 0].plot(self.metrics_history['success_rate'])
        axs[1, 0].set_title('Tỉ lệ thành công')
        axs[1, 0].set_xlabel('Bước')
        axs[1, 0].set_ylabel('Tỉ lệ')
        
        # 4. Mức độ tắc nghẽn
        axs[1, 1].plot(self.metrics_history['congestion'])
        axs[1, 1].set_title('Mức độ tắc nghẽn trung bình')
        axs[1, 1].set_xlabel('Bước')
        axs[1, 1].set_ylabel('Mức độ')
        
        plt.tight_layout()
        
        # Biểu đồ so sánh loại đồng thuận
        if hasattr(self, 'acsc') and hasattr(self.acsc, 'get_statistics'):
            stats = self.acsc.get_statistics()
            if 'strategy_usage' in stats:
                plt.figure(figsize=(10, 6))
                strategies = list(stats['strategy_usage'].keys())
                usage = list(stats['strategy_usage'].values())
                
                plt.bar(strategies, usage)
                plt.title('Phân bố loại đồng thuận')
                plt.xlabel('Chiến lược')
                plt.ylabel('Số lần sử dụng')
                
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
        """Lấy thống kê tổng hợp từ mô phỏng"""
        # Cấu hình mô phỏng
        sim_config = {
            "num_shards": len(self.network.shards) if hasattr(self.network, 'shards') else 0,
            "consensus_algorithm": getattr(self, 'consensus_algorithm', 'PoW'),
            "block_time": getattr(self, 'block_time', 10),
            "network_latency": getattr(self, 'network_latency', 100),
            "iterations": getattr(self, 'current_step', 100)
        }
        
        # Thống kê giao dịch
        if not hasattr(self, 'transactions_per_step'):
            self.transactions_per_step = getattr(self, 'tx_per_step', 20)
        
        transaction_stats = {
            "total": self.total_transactions,
            "successful": self.successful_transactions,
            "success_rate": self.successful_transactions / max(1, self.total_transactions),
            "transactions_per_step": self.transactions_per_step,
            "cross_shard_transactions": {
                "total": sum(1 for tx_data in self.transaction_data.values() if tx_data.get('is_cross_shard', False)),
                "successful": sum(1 for tx_data in self.transaction_data.values() 
                                if tx_data.get('is_cross_shard', False) and tx_data.get('status') == 'processed'),
                "success_rate": sum(1 for tx_data in self.transaction_data.values() 
                                    if tx_data.get('is_cross_shard', False) and tx_data.get('status') == 'processed') / 
                                max(1, sum(1 for tx_data in self.transaction_data.values() if tx_data.get('is_cross_shard', False)))
            }
        }
        
        # Thống kê hiệu suất
        performance_stats = {
            "avg_throughput": self.current_throughput if hasattr(self, 'current_throughput') else 0,
            "avg_latency": self.current_latency if hasattr(self, 'current_latency') else 0,
            "energy_consumption": self.current_energy if hasattr(self, 'current_energy') else 0,
            "congestion_level": self.current_congestion if hasattr(self, 'current_congestion') else 0
        }
        
        # Thống kê mô-đun
        module_stats = {}
        
        # MAD-RAPID nếu được bật
        if hasattr(self, 'mad_rapid') and self.mad_rapid is not None:
            madr_stats = {
                "optimized_tx_count": getattr(self.mad_rapid, 'optimized_tx_count', 0),
                "total_tx_processed": getattr(self.mad_rapid, 'total_tx_processed', 0),
                "avg_optimization_time": getattr(self.mad_rapid, 'avg_optimization_time', 0),
                "congestion_prediction_accuracy": getattr(self.mad_rapid, 'prediction_accuracy', 0)
            }
            module_stats["mad_rapid"] = madr_stats
        
        # ACSC nếu được bật
        if hasattr(self, 'acsc') and self.acsc is not None:
            acsc_stats = {
                "successful_tx_count": getattr(self.acsc, 'successful_tx_count', 0),
                "total_tx_processed": getattr(self.acsc, 'total_tx_processed', 0),
                "strategy_usage": getattr(self.acsc, 'strategy_usage', {})
            }
            module_stats["acsc"] = acsc_stats
        
        # Kết hợp tất cả thống kê
        summary_stats = {
            "simulation_config": sim_config,
            "transaction_stats": transaction_stats,
            "performance_stats": performance_stats,
            "module_stats": module_stats
        }
        
        return summary_stats

    def integrate_mad_rapid(self):
        """
        Kết nối MAD-RAPID với mô phỏng
        """
        if not hasattr(self, 'mad_rapid') or self.mad_rapid is None:
            # Nếu MAD-RAPID chưa được khởi tạo, khởi tạo lại
            from dqn_blockchain_sim.blockchain.mad_rapid import MADRAPIDProtocol
            
            # Khởi tạo MAD-RAPID
            self.mad_rapid = MADRAPIDProtocol(
                input_size=8,
                embedding_dim=64,
                hidden_size=128,
                lookback_window=10,
                prediction_horizon=5
            )
        
        # Kết nối MAD-RAPID với mạng blockchain
        self._hook_mad_rapid()

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

    def _hook_mad_rapid(self):
        """Kết nối MAD-RAPID với mạng blockchain"""
        import weakref
        print("Đang kết nối MAD-RAPID với mạng blockchain...")

        # Kiểm tra xem MAD-RAPID đã được tích hợp chưa
        if not hasattr(self, 'mad_rapid') or self.mad_rapid is None:
            print("MAD-RAPID chưa được khởi tạo, đang khởi tạo lại...")
            from dqn_blockchain_sim.blockchain.mad_rapid import MADRAPIDProtocol
            
            # Khởi tạo MAD-RAPID
            self.mad_rapid = MADRAPIDProtocol(
                input_size=8,
                embedding_dim=64,
                hidden_size=128,
                lookback_window=10,
                prediction_horizon=5
            )
            print("Đã khởi tạo MAD-RAPID: ", self.mad_rapid)
        
        # Thiết lập tham chiếu từ MAD-RAPID đến mô phỏng (sử dụng weakref để tránh vòng lặp tham chiếu)
        self.mad_rapid.simulation = weakref.proxy(self)
        
        # Thiết lập tham chiếu từ MAD-RAPID đến mạng
        self.mad_rapid.network = self.network
        
        # Khởi tạo số shard cho MAD-RAPID
        self.mad_rapid.num_shards = self.num_shards
        print(f"Số shard trong MAD-RAPID: {self.mad_rapid.num_shards}")
        
        # Khởi tạo các thuộc tính cần thiết cho các shard
        if not hasattr(self.mad_rapid, 'shard_states'):
            self.mad_rapid.shard_states = {}
        
        if not hasattr(self.mad_rapid, 'shard_agents'):
            self.mad_rapid.shard_agents = {}
            
        # Khởi tạo DQN agents cho mỗi shard nếu chưa có
        if self.use_dqn and (not hasattr(self.mad_rapid, 'dqn_agents') or not self.mad_rapid.dqn_agents):
            from dqn_blockchain_sim.agents.dqn_agent import ShardDQNAgent
            
            # Khởi tạo DQN agents cho mỗi shard
            self.mad_rapid.dqn_agents = {}
            for shard_id in range(self.num_shards):
                # Khởi tạo agent với state_size=8 (đặc trưng shard) và action_size=3 (các hành động tối ưu)
                agent = ShardDQNAgent(
                    shard_id=shard_id,
                    state_size=8,
                    action_size=3
                )
                # Gán shard cho agent
                if shard_id < len(self.network.shards):
                    agent.shard = self.network.shards[shard_id]
                
                # Lưu agent vào danh sách
                self.mad_rapid.dqn_agents[shard_id] = agent
                
            print(f"Đã khởi tạo {len(self.mad_rapid.dqn_agents)} DQN agents cho MAD-RAPID")
            
        # Khởi tạo các thuộc tính theo dõi hiệu suất
        if not hasattr(self.mad_rapid, 'transaction_history'):
            self.mad_rapid.transaction_history = []
            
        if not hasattr(self.mad_rapid, 'optimization_history'):
            self.mad_rapid.optimization_history = []
            
        if not hasattr(self.mad_rapid, 'cross_shard_success_count'):
            self.mad_rapid.cross_shard_success_count = 0
        
        # Khởi tạo các thuộc tính cần thiết cho các shard
        for shard_id, shard in self.network.shards.items():
            # Đảm bảo rằng mỗi shard có các thuộc tính cần thiết
            for attr in ['transaction_queue', 'cross_shard_queue', 'processed_transactions',
                        'failed_transactions', 'confirmed_transactions', 'rejected_transactions',
                        'throughput', 'latency', 'congestion_level', 'cross_shard_tx_count',
                        'total_gas_used', 'total_fees', 'successful_tx_count', 'successful_cs_tx_count']:
                if not hasattr(shard, attr):
                    if attr in ['transaction_queue', 'cross_shard_queue']:
                        setattr(shard, attr, [])
                    elif attr in ['processed_transactions', 'failed_transactions', 'confirmed_transactions', 'rejected_transactions']:
                        setattr(shard, attr, {})
                    elif attr in ['throughput', 'latency', 'congestion_level', 'cross_shard_tx_count',
                               'total_gas_used', 'total_fees', 'successful_tx_count', 'successful_cs_tx_count']:
                        setattr(shard, attr, 0)
        
        # Khởi tạo embeddings cho các shard
        if not hasattr(self.mad_rapid, 'shard_embeddings') or not self.mad_rapid.shard_embeddings:
            print("Đang khởi tạo shard_embeddings...")
            self.mad_rapid._initialize_shard_embeddings()
        
        # Ghi đè phương thức xử lý giao dịch xuyên shard trong các shard
        for shard_id, shard in self.network.shards.items():
            # Lưu lại phương thức gốc để sử dụng khi cần
            if not hasattr(shard, '_original_process_cross_shard'):
                shard._original_process_cross_shard = shard.process_cross_shard_transaction if hasattr(shard, 'process_cross_shard_transaction') else None
            
            # Ghi đè phương thức xử lý giao dịch xuyên shard
            def mad_rapid_process_tx(tx, shard=shard):
                # Sử dụng MAD-RAPID để xử lý giao dịch
                result = self.mad_rapid.process_cross_shard_transaction(tx)
                return result
            
            # Gán phương thức mới
            shard.process_cross_shard_transaction = mad_rapid_process_tx
            
        print("Đã kết nối MAD-RAPID với mạng blockchain thành công!")


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