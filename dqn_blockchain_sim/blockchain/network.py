"""
Module cung cấp các thành phần mạng blockchain
"""

import time
import uuid
import random
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any

from dqn_blockchain_sim.blockchain.shard import Shard
from dqn_blockchain_sim.blockchain.transaction import Transaction, TransactionStatus
from dqn_blockchain_sim.configs.simulation_config import BLOCKCHAIN_CONFIG


class BlockchainNetwork:
    """
    Lớp đại diện cho mạng blockchain hoàn chỉnh
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo mạng blockchain
        
        Args:
            config: Cấu hình mạng, sử dụng mặc định nếu không cung cấp
        """
        self.config = config if config is not None else BLOCKCHAIN_CONFIG
        self.shards: Dict[int, Shard] = {}
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Thông tin về từng nút
        self.node_to_shard: Dict[str, int] = {}  # Ánh xạ từ nút đến shard
        
        # Biểu đồ kết nối giữa các shard
        self.shard_graph = nx.Graph()
        
        # Thống kê hiệu suất mạng
        self.start_time = time.time()
        self.total_transactions = 0
        self.confirmed_transactions = 0
        self.cross_shard_transactions = 0
        self.total_energy_consumption = 0.0
        self.avg_network_latency = 0.0
        self.reshard_events = []
        
        # Khởi tạo các shard và nút
        self._initialize_network()
        
    def _initialize_network(self) -> None:
        """
        Khởi tạo cấu trúc mạng với các shard và nút
        """
        # Tạo các shard
        for i in range(self.config["num_shards"]):
            self.shards[i] = Shard(i, self, self.config["min_nodes_per_shard"], self.config["block_time"])
            self.shard_graph.add_node(i)
        
        # Kết nối các shard
        # Mỗi shard kết nối với một số shard khác để tạo thành mạng liên kết
        for i in range(self.config["num_shards"]):
            # Số lượng kết nối ngẫu nhiên cho mỗi shard
            num_connections = min(3, self.config["num_shards"] - 1)
            other_shards = list(set(range(self.config["num_shards"])) - {i})
            for j in random.sample(other_shards, num_connections):
                if not self.shard_graph.has_edge(i, j):
                    # Độ trễ ngẫu nhiên giữa các shard (ms)
                    latency = random.uniform(50, 200)
                    self.shard_graph.add_edge(i, j, latency=latency)
        
        # Tạo các nút và phân bổ vào các shard
        nodes_per_shard = self.config["num_nodes"] // self.config["num_shards"]
        remaining_nodes = self.config["num_nodes"] % self.config["num_shards"]
        
        node_id = 0
        for shard_id in range(self.config["num_shards"]):
            # Số lượng nút cho shard hiện tại
            shard_nodes = nodes_per_shard + (1 if shard_id < remaining_nodes else 0)
            
            for _ in range(shard_nodes):
                node_name = f"node_{node_id}"
                self.nodes[node_name] = {
                    'id': node_name,
                    'compute_power': random.uniform(0.5, 2.0),  # Năng lực tính toán tương đối
                    'reliability': random.uniform(0.8, 1.0),    # Độ tin cậy (0-1)
                    'energy_efficiency': random.uniform(0.6, 1.0),  # Hiệu suất năng lượng (0-1)
                    'uptime': 1.0,                              # Thời gian hoạt động (0-1)
                    'security_score': random.uniform(0.7, 1.0), # Điểm bảo mật (0-1)
                    'timestamp': time.time()                    # Thời điểm thêm vào mạng
                }
                self.node_to_shard[node_name] = shard_id
                self.shards[shard_id].add_node(node_name)
                node_id += 1
    
    def get_shard(self, shard_id: int) -> Optional[Shard]:
        """
        Lấy tham chiếu đến một shard cụ thể
        
        Args:
            shard_id: ID của shard cần lấy
            
        Returns:
            Đối tượng Shard tương ứng hoặc None nếu không tồn tại
        """
        return self.shards.get(shard_id)
    
    def get_node_shard(self, node_id: str) -> Optional[int]:
        """
        Lấy ID shard chứa một nút cụ thể
        
        Args:
            node_id: ID của nút cần kiểm tra
            
        Returns:
            ID của shard chứa nút hoặc None nếu nút không tồn tại
        """
        return self.node_to_shard.get(node_id)
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Thêm một giao dịch vào mạng
        
        Args:
            transaction: Giao dịch cần thêm
            
        Returns:
            True nếu thêm thành công, False nếu không
        """
        sender_shard_id = self.get_node_shard(transaction.sender_id)
        
        if sender_shard_id is None:
            return False
            
        sender_shard = self.get_shard(sender_shard_id)
        result = sender_shard.add_transaction(transaction)
        
        if result:
            self.total_transactions += 1
            
        return result
    
    def move_node(self, node_id: str, target_shard_id: int) -> bool:
        """
        Di chuyển một nút từ shard hiện tại sang shard mới
        
        Args:
            node_id: ID của nút cần di chuyển
            target_shard_id: ID của shard đích
            
        Returns:
            True nếu di chuyển thành công, False nếu không
        """
        current_shard_id = self.get_node_shard(node_id)
        
        if (current_shard_id is None or 
            target_shard_id not in self.shards or 
            current_shard_id == target_shard_id):
            return False
            
        current_shard = self.get_shard(current_shard_id)
        target_shard = self.get_shard(target_shard_id)
        
        # Kiểm tra điều kiện di chuyển
        if (len(current_shard.nodes) <= current_shard.min_nodes):
            return False
            
        # Xóa khỏi shard hiện tại
        if not current_shard.remove_node(node_id):
            return False
            
        # Thêm vào shard mới
        if not target_shard.add_node(node_id):
            # Nếu không thể thêm vào shard mới, thêm lại vào shard cũ
            current_shard.add_node(node_id)
            return False
            
        # Cập nhật ánh xạ nút
        self.node_to_shard[node_id] = target_shard_id
        
        return True
    
    def optimize_shards(self) -> List[Dict[str, Any]]:
        """
        Tối ưu hóa phân phối nút giữa các shard dựa trên tải
        
        Returns:
            Danh sách các thay đổi đã thực hiện
        """
        changes = []
        
        # Tính toán mức tải trung bình cho từng shard
        shard_loads = {}
        for shard_id, shard in self.shards.items():
            shard_loads[shard_id] = shard.congestion_level
            
        # Xác định shard quá tải và shard ít tải
        avg_load = sum(shard_loads.values()) / len(shard_loads)
        overloaded_shards = [sid for sid, load in shard_loads.items() if load > avg_load * 1.2]
        underloaded_shards = [sid for sid, load in shard_loads.items() if load < avg_load * 0.8]
        
        if not overloaded_shards or not underloaded_shards:
            return changes
            
        # Di chuyển nút từ shard quá tải sang shard ít tải
        max_moves = self.config["max_reshard_nodes"]
        moves_count = 0
        
        for src_id in overloaded_shards:
            if moves_count >= max_moves:
                break
                
            src_shard = self.get_shard(src_id)
            
            # Đảm bảo shard nguồn có đủ nút để di chuyển
            if len(src_shard.nodes) <= src_shard.min_nodes:
                continue
                
            for dst_id in underloaded_shards:
                if moves_count >= max_moves:
                    break
                    
                # Chọn ngẫu nhiên một nút để di chuyển
                movable_nodes = list(src_shard.nodes)
                random.shuffle(movable_nodes)
                
                for node_id in movable_nodes:
                    if self.move_node(node_id, dst_id):
                        changes.append({
                            'node_id': node_id,
                            'from_shard': src_id,
                            'to_shard': dst_id,
                            'timestamp': time.time()
                        })
                        moves_count += 1
                        break
        
        if changes:
            self.reshard_events.append({
                'timestamp': time.time(),
                'changes': changes,
                'reason': 'load_balancing'
            })
            
        return changes
    
    def process_network(self, current_time: float) -> Dict[str, Any]:
        """
        Xử lý toàn bộ mạng blockchain cho một bước thời gian
        
        Args:
            current_time: Thời gian hiện tại
            
        Returns:
            Thống kê về trạng thái mạng
        """
        total_processed = 0
        
        # Xử lý giao dịch trong mỗi shard
        for shard_id, shard in self.shards.items():
            processed = shard.process_transactions(current_time)
            total_processed += processed
            
            # Cập nhật thống kê mạng
            self.confirmed_transactions += shard.confirmed_transactions
            self.cross_shard_transactions += shard.cross_shard_transactions
            self.total_energy_consumption += shard.energy_consumption
            
        # Kiểm tra xem có cần tối ưu hóa phân phối nút không
        should_reshard = False
        
        # Kiểm tra các điều kiện tái phân mảnh
        for shard_id, shard in self.shards.items():
            if shard.congestion_level > self.config["reshard_threshold"]:
                should_reshard = True
                break
                
        # Thực hiện tái phân mảnh nếu cần và nếu đã qua thời gian cooldown
        last_reshard_time = self.reshard_events[-1]["timestamp"] if self.reshard_events else 0
        if should_reshard and (current_time - last_reshard_time) > self.config["reshard_cooldown"]:
            changes = self.optimize_shards()
            
        # Cập nhật độ trễ trung bình mạng
        total_latency = 0
        latency_count = 0
        
        for shard in self.shards.values():
            if shard.confirmed_transactions > 0:
                total_latency += shard.avg_latency * shard.confirmed_transactions
                latency_count += shard.confirmed_transactions
                
        if latency_count > 0:
            self.avg_network_latency = total_latency / latency_count
            
        # Thu thập thống kê
        stats = self.get_network_stats()
        
        return stats
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất mạng
        
        Returns:
            Từ điển chứa các thống kê mạng
        """
        # Thu thập thống kê từ từng shard
        shard_stats = {shard_id: shard.get_stats() for shard_id, shard in self.shards.items()}
        
        return {
            'timestamp': time.time(),
            'runtime': time.time() - self.start_time,
            'num_shards': len(self.shards),
            'num_nodes': len(self.nodes),
            'total_transactions': self.total_transactions,
            'confirmed_transactions': self.confirmed_transactions,
            'cross_shard_transactions': self.cross_shard_transactions,
            'avg_network_latency': self.avg_network_latency,
            'total_energy_consumption': self.total_energy_consumption,
            'reshard_count': len(self.reshard_events),
            'shard_stats': shard_stats,
            'network_congestion': np.mean([shard.congestion_level for shard in self.shards.values()])
        }
    
    def generate_random_transaction(self) -> Transaction:
        """
        Tạo một giao dịch ngẫu nhiên cho mục đích kiểm thử
        
        Returns:
            Giao dịch ngẫu nhiên mới
        """
        # Chọn ngẫu nhiên người gửi và người nhận
        all_nodes = list(self.nodes.keys())
        sender = random.choice(all_nodes)
        receiver = random.choice(all_nodes)
        
        # Tránh gửi cho chính mình
        while receiver == sender:
            receiver = random.choice(all_nodes)
            
        # Tạo giao dịch với giá trị ngẫu nhiên
        amount = random.uniform(0.1, 100.0)
        
        # Dữ liệu bổ sung ngẫu nhiên
        data_size = random.randint(0, 5)  # Độ phức tạp của dữ liệu
        data = {}
        
        if data_size > 0:
            for i in range(data_size):
                data[f"field_{i}"] = f"value_{random.randint(0, 1000)}"
                
        gas_price = random.uniform(0.5, 2.0)
        gas_limit = random.randint(21000, 100000)
        
        return Transaction(sender, receiver, amount, data, gas_price, gas_limit)
    
    def simulate_attack(self, attack_type: str, intensity: float) -> Dict[str, Any]:
        """
        Mô phỏng một cuộc tấn công vào mạng
        
        Args:
            attack_type: Loại tấn công ('sybil', 'ddos', 'eclipse', v.v.)
            intensity: Cường độ tấn công (0-1)
            
        Returns:
            Thông tin về tấn công đã mô phỏng
        """
        attack_info = {
            'type': attack_type,
            'intensity': intensity,
            'timestamp': time.time(),
            'affected_shards': [],
            'affected_nodes': []
        }
        
        if attack_type == 'sybil':
            # Mô phỏng tấn công Sybil bằng cách giảm độ tin cậy của một số nút
            num_affected = int(len(self.nodes) * intensity)
            affected_nodes = random.sample(list(self.nodes.keys()), num_affected)
            
            for node_id in affected_nodes:
                # Giảm độ tin cậy của nút
                self.nodes[node_id]['reliability'] *= 0.5
                self.nodes[node_id]['security_score'] *= 0.3
                
                # Thêm vào danh sách bị ảnh hưởng
                attack_info['affected_nodes'].append(node_id)
                shard_id = self.get_node_shard(node_id)
                if shard_id not in attack_info['affected_shards']:
                    attack_info['affected_shards'].append(shard_id)
                
        elif attack_type == 'ddos':
            # Tấn công DDoS - tăng tình trạng tắc nghẽn của một số shard
            num_affected = max(1, int(len(self.shards) * intensity))
            affected_shards = random.sample(list(self.shards.keys()), num_affected)
            
            for shard_id in affected_shards:
                # Tăng mức độ tắc nghẽn
                self.shards[shard_id].congestion_level = min(1.0, self.shards[shard_id].congestion_level + 0.5)
                
                # Thêm vào các thông tin bị ảnh hưởng
                attack_info['affected_shards'].append(shard_id)
                for node_id in self.shards[shard_id].nodes:
                    attack_info['affected_nodes'].append(node_id)
                    
        return attack_info
    
    def get_all_trust_scores(self) -> Dict[str, float]:
        """
        Lấy điểm tin cậy của tất cả các nút
        
        Returns:
            Từ điển ánh xạ từ node_id đến điểm tin cậy
        """
        return self.current_trust_scores.copy()
        
    def reset(self) -> None:
        """
        Đặt lại trạng thái mạng blockchain
        """
        # Xóa tất cả các shard hiện tại
        self.shards = {}
        self.nodes = {}
        self.node_to_shard = {}
        
        # Đặt lại thống kê
        self.start_time = time.time()
        self.total_transactions = 0
        self.confirmed_transactions = 0
        self.cross_shard_transactions = 0
        self.total_energy_consumption = 0.0
        self.avg_network_latency = 0.0
        self.reshard_events = []
        
        # Khởi tạo lại mạng
        self._initialize_network() 