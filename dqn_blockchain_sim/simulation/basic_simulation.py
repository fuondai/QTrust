"""
Module mô phỏng blockchain cơ bản cho việc benchmark
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

from dqn_blockchain_sim.blockchain.network import BlockchainNetwork
from dqn_blockchain_sim.blockchain.consensus import ConsensusProtocol
from dqn_blockchain_sim.utils.transaction_generator import generate_transactions

class BasicSimulation:
    """
    Lớp mô phỏng cơ bản cho việc benchmark các giao thức đồng thuận
    """
    
    def __init__(self, 
                num_shards: int = 4,
                consensus_protocol: Optional[ConsensusProtocol] = None,
                config: Dict[str, Any] = None):
        """
        Khởi tạo mô phỏng cơ bản
        
        Args:
            num_shards: Số lượng shard
            consensus_protocol: Giao thức đồng thuận sử dụng
            config: Cấu hình cho mô phỏng
        """
        self.num_shards = num_shards
        self.consensus_protocol = consensus_protocol
        self.config = config if config else {}
        
        # Khởi tạo mạng blockchain
        self.network = BlockchainNetwork(num_shards=num_shards)
        
        # Khởi tạo các biến theo dõi
        self.total_transactions = 0
        self.successful_transactions = 0
        self.failed_transactions = 0
        
        # Lưu trữ metrics
        self.metrics_history = defaultdict(list)
        
    def reset(self):
        """
        Đặt lại trạng thái mô phỏng
        """
        self.network = BlockchainNetwork(num_shards=self.num_shards)
        self.total_transactions = 0
        self.successful_transactions = 0
        self.failed_transactions = 0
        self.metrics_history = defaultdict(list)
        
        if self.consensus_protocol:
            self.consensus_protocol.reset_statistics()
            
    def step(self, transactions: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Thực hiện một bước mô phỏng
        
        Args:
            transactions: Danh sách giao dịch cần xử lý
            
        Returns:
            Thống kê của bước hiện tại
        """
        if transactions is None:
            transactions = generate_transactions(5)  # Mặc định 5 giao dịch/bước
            
        # Xử lý từng giao dịch
        for tx in transactions:
            self.total_transactions += 1
            
            # Xác định shard xử lý
            shard_id = tx.get('shard_id', hash(str(tx)) % self.num_shards)
            
            # Xử lý giao dịch bằng giao thức đồng thuận
            if self.consensus_protocol:
                success, latency, energy = self.consensus_protocol.process_transaction(tx)
            else:
                # Xử lý mặc định nếu không có giao thức đồng thuận
                success = np.random.random() > 0.1  # 90% thành công
                latency = np.random.normal(50, 10)  # Độ trễ trung bình 50ms
                energy = np.random.normal(1, 0.2)   # Năng lượng trung bình 1 đơn vị
                
            if success:
                self.successful_transactions += 1
            else:
                self.failed_transactions += 1
                
            # Cập nhật trạng thái shard
            shard = self.network.shards[shard_id]
            shard.process_transaction(tx)
            
        # Cập nhật và trả về thống kê
        return self._update_metrics()
        
    def _update_metrics(self) -> Dict[str, Any]:
        """
        Cập nhật và trả về các metric hiện tại
        
        Returns:
            Từ điển chứa các metric
        """
        # Tính toán các metric cơ bản
        success_rate = self.successful_transactions / max(1, self.total_transactions)
        
        # Tính toán các metric trung bình trên các shard
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
        
        # Lấy thống kê từ giao thức đồng thuận
        consensus_stats = {}
        if self.consensus_protocol:
            consensus_stats = self.consensus_protocol.get_statistics()
            
        # Tạo từ điển metric
        metrics = {
            'timestamp': time.time(),
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'failed_transactions': self.failed_transactions,
            'success_rate': success_rate,
            'avg_throughput': avg_throughput,
            'avg_latency': avg_latency,
            'avg_congestion': avg_congestion,
            'consensus_stats': consensus_stats
        }
        
        # Cập nhật lịch sử
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.metrics_history[key].append(value)
                
        return metrics
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê tổng hợp của toàn bộ mô phỏng
        
        Returns:
            Từ điển chứa các thống kê
        """
        stats = {
            'total_transactions': self.total_transactions,
            'successful_transactions': self.successful_transactions,
            'failed_transactions': self.failed_transactions,
            'success_rate': self.successful_transactions / max(1, self.total_transactions),
            'metrics_history': dict(self.metrics_history)
        }
        
        if self.consensus_protocol:
            stats['consensus_stats'] = self.consensus_protocol.get_statistics()
            
        return stats 