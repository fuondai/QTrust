"""
Module mô phỏng phân mảnh (shard) trong blockchain
"""

import time
import uuid
from typing import Dict, List, Optional, Set, Tuple, Any
import random
import networkx as nx

from dqn_blockchain_sim.blockchain.transaction import Transaction, TransactionStatus


class Shard:
    """
    Lớp mô phỏng một phân mảnh (shard) trong mạng blockchain
    """
    
    def __init__(self, 
                 shard_id: int, 
                 network, 
                 min_nodes: int = 5,
                 block_time: int = 10):
        """
        Khởi tạo một shard mới
        
        Args:
            shard_id: ID của shard
            network: Tham chiếu đến mạng blockchain chứa shard
            min_nodes: Số lượng nút tối thiểu trong shard
            block_time: Thời gian tạo block (giây)
        """
        self.shard_id = shard_id
        self.network = network
        self.nodes: Set[str] = set()  # Tập hợp các nút trong shard
        self.min_nodes = min_nodes
        self.block_time = block_time
        self.transaction_pool: Dict[str, Transaction] = {}  # Pool chứa các giao dịch đang xử lý
        self.processed_transactions: Dict[str, Transaction] = {}  # Các giao dịch đã xử lý
        self.cross_shard_queue: List[Transaction] = []  # Hàng đợi giao dịch xuyên mảnh
        
        # Thống kê hiệu suất
        self.total_transactions = 0
        self.confirmed_transactions = 0
        self.rejected_transactions = 0
        self.cross_shard_transactions = 0
        self.avg_latency = 0.0
        self.energy_consumption = 0.0
        
        # Thông tin về trạng thái tải và mức độ tắc nghẽn
        self.congestion_level = 0.0  # Từ 0 (không tắc nghẽn) đến 1 (tắc nghẽn hoàn toàn)
        self.load_history: List[float] = []
        self.last_block_timestamp = time.time()
        self.current_block_transactions: List[Transaction] = []
        
        # Biểu đồ kết nối giữa các nút trong shard
        self.node_graph = nx.Graph()
        
    def add_node(self, node_id: str) -> bool:
        """
        Thêm một nút vào shard
        
        Args:
            node_id: ID của nút cần thêm
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        if node_id in self.nodes:
            return False
            
        self.nodes.add(node_id)
        self.node_graph.add_node(node_id)
        
        # Tạo liên kết với một số nút ngẫu nhiên trong shard
        if len(self.nodes) > 1:
            num_connections = min(3, len(self.nodes) - 1)
            for other_node in random.sample(list(self.nodes - {node_id}), num_connections):
                self.node_graph.add_edge(node_id, other_node, latency=random.uniform(10, 100))
                
        return True
    
    def remove_node(self, node_id: str) -> bool:
        """
        Loại bỏ một nút khỏi shard
        
        Args:
            node_id: ID của nút cần loại bỏ
            
        Returns:
            True nếu loại bỏ thành công, False nếu không
        """
        if node_id not in self.nodes or len(self.nodes) <= self.min_nodes:
            return False
            
        self.nodes.remove(node_id)
        self.node_graph.remove_node(node_id)
        return True
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Thêm một giao dịch vào shard để xử lý
        
        Args:
            transaction: Giao dịch cần xử lý
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        if transaction.transaction_id in self.transaction_pool or transaction.transaction_id in self.processed_transactions:
            return False
            
        transaction.source_shard = self.shard_id
        
        # Xác định target shard
        # (Trong triển khai thực, sẽ dựa vào thuật toán định tuyến phức tạp hơn)
        receiver_shard = self.network.get_node_shard(transaction.receiver_id)
        transaction.target_shard = receiver_shard
        
        # Kiểm tra nếu là giao dịch xuyên mảnh
        if transaction.is_cross_shard():
            transaction.update_status(TransactionStatus.CROSS_SHARD)
            self.cross_shard_queue.append(transaction)
            self.cross_shard_transactions += 1
            return True
            
        # Xử lý thông thường cho giao dịch trong cùng shard
        self.transaction_pool[transaction.transaction_id] = transaction
        self.total_transactions += 1
        
        # Cập nhật mức độ tắc nghẽn
        self.update_congestion_level()
        
        return True
    
    def process_transactions(self, current_time: float, max_txs: int = 100) -> int:
        """
        Xử lý các giao dịch trong pool
        
        Args:
            current_time: Thời gian hiện tại
            max_txs: Số lượng giao dịch tối đa xử lý mỗi lần
            
        Returns:
            Số lượng giao dịch đã xử lý
        """
        if not self.transaction_pool:
            return 0
            
        processed_count = 0
        txs_to_process = min(max_txs, len(self.transaction_pool))
        tx_ids = list(self.transaction_pool.keys())[:txs_to_process]
        
        for tx_id in tx_ids:
            tx = self.transaction_pool[tx_id]
            
            # Kiểm tra timeout
            if current_time - tx.timestamp > self.network.config["transaction_timeout"]:
                tx.update_status(TransactionStatus.TIMEOUT)
                self.processed_transactions[tx_id] = tx
                del self.transaction_pool[tx_id]
                self.rejected_transactions += 1
                continue
                
            # Mô phỏng xác thực giao dịch
            validators_needed = max(1, int(len(self.nodes) * self.network.config["verification_percentage"]))
            tx.required_confirmations = validators_needed
            
            # Chọn ngẫu nhiên các validator từ các nút trong shard
            validators = random.sample(list(self.nodes), min(validators_needed, len(self.nodes)))
            
            # Mô phỏng việc xác thực
            all_confirmed = True
            for validator in validators:
                # Xác suất xác thực thành công (có thể điều chỉnh dựa trên độ tin cậy của nút)
                if random.random() < 0.95:  # 95% xác suất xác thực thành công
                    tx.add_confirmation(validator)
                else:
                    all_confirmed = False
                    
            # Tiêu thụ năng lượng cho việc xác thực
            energy_per_validation = 0.001  # kWh, giá trị mô phỏng
            self.energy_consumption += len(validators) * energy_per_validation
                    
            if all_confirmed:
                tx.update_status(TransactionStatus.CONFIRMED)
                self.confirmed_transactions += 1
                # Cập nhật độ trễ trung bình
                new_latency = tx.get_latency()
                if new_latency > 0:
                    self.avg_latency = (self.avg_latency * (self.confirmed_transactions - 1) + new_latency) / self.confirmed_transactions
                
            # Nếu đã xác thực hoặc bị từ chối, chuyển sang danh sách đã xử lý
            if tx.status in [TransactionStatus.CONFIRMED, TransactionStatus.REJECTED]:
                self.processed_transactions[tx_id] = tx
                del self.transaction_pool[tx_id]
                self.current_block_transactions.append(tx)
                processed_count += 1
                
        # Kiểm tra xem có nên tạo block mới không
        if current_time - self.last_block_timestamp >= self.block_time and self.current_block_transactions:
            self.create_new_block()
            
        # Xử lý giao dịch xuyên mảnh
        self.process_cross_shard_transactions()
            
        return processed_count
    
    def process_cross_shard_transactions(self) -> int:
        """
        Xử lý các giao dịch xuyên mảnh
        
        Returns:
            Số lượng giao dịch xuyên mảnh đã xử lý
        """
        if not self.cross_shard_queue:
            return 0
            
        processed_count = 0
        for tx in self.cross_shard_queue[:]:
            target_shard = self.network.get_shard(tx.target_shard)
            if target_shard:
                # Chuyển giao dịch đến shard đích
                if target_shard.add_transaction(tx):
                    self.cross_shard_queue.remove(tx)
                    processed_count += 1
                    
        return processed_count
    
    def create_new_block(self) -> Dict[str, Any]:
        """
        Tạo block mới từ các giao dịch đã xác thực
        
        Returns:
            Thông tin về block mới
        """
        if not self.current_block_transactions:
            return {}
            
        block = {
            'block_id': str(uuid.uuid4()),
            'shard_id': self.shard_id,
            'timestamp': time.time(),
            'previous_block': self.last_block_timestamp,
            'transaction_count': len(self.current_block_transactions),
            'transactions': [tx.transaction_id for tx in self.current_block_transactions],
            'energy_used': self.energy_consumption  # Năng lượng tiêu thụ từ lần tạo block cuối
        }
        
        # Cập nhật trạng thái
        self.last_block_timestamp = time.time()
        self.current_block_transactions = []
        
        return block
    
    def update_congestion_level(self) -> None:
        """
        Cập nhật mức độ tắc nghẽn dựa trên số lượng giao dịch trong pool
        """
        # Giả sử mỗi nút có thể xử lý tối đa 10 giao dịch mỗi giây
        max_capacity = len(self.nodes) * 10
        current_load = len(self.transaction_pool)
        
        self.congestion_level = min(1.0, current_load / max(1, max_capacity))
        self.load_history.append(self.congestion_level)
        
        # Chỉ giữ lịch sử tải gần đây (5 phút)
        max_history_len = 300  # 5 phút với giả định cập nhật mỗi giây
        if len(self.load_history) > max_history_len:
            self.load_history = self.load_history[-max_history_len:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất của shard
        
        Returns:
            Từ điển chứa các thống kê hiệu suất
        """
        return {
            'shard_id': self.shard_id,
            'node_count': len(self.nodes),
            'total_transactions': self.total_transactions,
            'confirmed_transactions': self.confirmed_transactions,
            'rejected_transactions': self.rejected_transactions,
            'cross_shard_transactions': self.cross_shard_transactions,
            'pending_transactions': len(self.transaction_pool),
            'avg_latency': self.avg_latency,
            'congestion_level': self.congestion_level,
            'energy_consumption': self.energy_consumption,
            'avg_load': sum(self.load_history) / max(1, len(self.load_history)) if self.load_history else 0
        }
    
    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của shard
        
        Returns:
            Chuỗi mô tả shard
        """
        return f"Shard(id={self.shard_id}, nodes={len(self.nodes)}, txs_pending={len(self.transaction_pool)}, congestion={self.congestion_level:.2f})" 