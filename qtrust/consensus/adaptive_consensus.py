import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import random

class ConsensusProtocol:
    """
    Lớp cơ sở cho các giao thức đồng thuận.
    """
    def __init__(self, name: str, latency_factor: float, energy_factor: float, security_factor: float):
        """
        Khởi tạo giao thức đồng thuận.
        
        Args:
            name: Tên giao thức
            latency_factor: Hệ số độ trễ (1.0 = độ trễ cơ sở)
            energy_factor: Hệ số tiêu thụ năng lượng (1.0 = năng lượng cơ sở)
            security_factor: Hệ số bảo mật (1.0 = bảo mật tối đa)
        """
        self.name = name
        self.latency_factor = latency_factor
        self.energy_factor = energy_factor
        self.security_factor = security_factor
    
    def execute(self, transaction: Dict[str, Any], nodes: List[int], trust_scores: List[float]) -> Tuple[bool, float, float]:
        """
        Thực hiện giao thức đồng thuận trên một giao dịch.
        
        Args:
            transaction: Giao dịch cần được xác nhận
            nodes: Danh sách các node tham gia vào quá trình đồng thuận
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")

class FastBFT(ConsensusProtocol):
    """
    Fast Byzantine Fault Tolerance - Giao thức đồng thuận nhanh với độ bảo mật thấp hơn.
    Thích hợp cho các giao dịch có giá trị thấp và yêu cầu xử lý nhanh.
    """
    def __init__(self):
        super(FastBFT, self).__init__(
            name="Fast_BFT",
            latency_factor=0.5,    # Độ trễ thấp
            energy_factor=0.6,     # Tiêu thụ năng lượng thấp
            security_factor=0.7    # Bảo mật vừa phải
        )
    
    def execute(self, transaction: Dict[str, Any], nodes: List[int], trust_scores: List[float]) -> Tuple[bool, float, float]:
        """
        Thực hiện Fast BFT.
        
        Args:
            transaction: Giao dịch cần được xác nhận
            nodes: Danh sách các node tham gia vào quá trình đồng thuận
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Số lượng node tối thiểu cần để đạt đồng thuận (1/2 tổng số node)
        min_nodes_required = len(nodes) // 2 + 1
        
        # Mô phỏng quá trình đồng thuận
        # Chọn ngẫu nhiên các node tham gia biểu quyết, có trọng số theo điểm tin cậy
        voting_nodes = random.sample(range(len(nodes)), min(len(nodes), min_nodes_required + 2))
        
        # Tính điểm tin cậy trung bình của các node biểu quyết
        avg_trust = np.mean([trust_scores[i] for i in voting_nodes])
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình
        success_prob = min(0.98, avg_trust * 0.9 + 0.1)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = random.random() < success_prob
        
        # Tính độ trễ
        # Fast BFT có độ trễ thấp nhưng vẫn phụ thuộc vào số lượng node và giá trị giao dịch
        latency = self.latency_factor * (0.5 + 0.2 * len(voting_nodes) / len(nodes))
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (5.0 + 0.5 * len(voting_nodes))
        
        return consensus_achieved, latency, energy

class PBFT(ConsensusProtocol):
    """
    Practical Byzantine Fault Tolerance - Giao thức đồng thuận cân bằng giữa hiệu suất và bảo mật.
    Phù hợp với hầu hết các giao dịch thông thường.
    """
    def __init__(self):
        super(PBFT, self).__init__(
            name="PBFT",
            latency_factor=1.0,    # Độ trễ trung bình
            energy_factor=1.0,     # Tiêu thụ năng lượng trung bình
            security_factor=0.85   # Bảo mật cao
        )
    
    def execute(self, transaction: Dict[str, Any], nodes: List[int], trust_scores: List[float]) -> Tuple[bool, float, float]:
        """
        Thực hiện PBFT.
        
        Args:
            transaction: Giao dịch cần được xác nhận
            nodes: Danh sách các node tham gia vào quá trình đồng thuận
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Số lượng node tối thiểu cần để đạt đồng thuận (2/3 tổng số node)
        min_nodes_required = 2 * len(nodes) // 3 + 1
        
        # Mô phỏng quá trình đồng thuận
        # PBFT yêu cầu nhiều node hơn để đạt đồng thuận
        voting_nodes = random.sample(range(len(nodes)), min(len(nodes), min_nodes_required))
        
        # Tính điểm tin cậy trung bình của các node biểu quyết
        avg_trust = np.mean([trust_scores[i] for i in voting_nodes])
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình và số lượng node
        success_prob = min(0.99, avg_trust * 0.95 + 0.05)
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = random.random() < success_prob
        
        # Tính độ trễ
        # PBFT có độ trễ trung bình và phụ thuộc nhiều vào số lượng node
        latency = self.latency_factor * (1.0 + 0.3 * len(voting_nodes) / len(nodes))
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (10.0 + 1.0 * len(voting_nodes))
        
        return consensus_achieved, latency, energy

class RobustBFT(ConsensusProtocol):
    """
    Robust Byzantine Fault Tolerance - Giao thức đồng thuận với độ bảo mật cao nhất.
    Thích hợp cho các giao dịch có giá trị cao và yêu cầu bảo mật tối đa.
    """
    def __init__(self):
        super(RobustBFT, self).__init__(
            name="Robust_BFT",
            latency_factor=2.0,    # Độ trễ cao
            energy_factor=1.5,     # Tiêu thụ năng lượng cao
            security_factor=1.0    # Bảo mật tối đa
        )
    
    def execute(self, transaction: Dict[str, Any], nodes: List[int], trust_scores: List[float]) -> Tuple[bool, float, float]:
        """
        Thực hiện Robust BFT.
        
        Args:
            transaction: Giao dịch cần được xác nhận
            nodes: Danh sách các node tham gia vào quá trình đồng thuận
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Số lượng node tối thiểu cần để đạt đồng thuận (3/4 tổng số node)
        min_nodes_required = 3 * len(nodes) // 4 + 1
        
        # Mô phỏng quá trình đồng thuận
        # Robust BFT yêu cầu rất nhiều node để đạt đồng thuận
        voting_nodes = random.sample(range(len(nodes)), min(len(nodes), min_nodes_required))
        
        # Lọc ra các node có điểm tin cậy cao
        trusted_nodes = [i for i in voting_nodes if trust_scores[i] > 0.7]
        
        # Tính điểm tin cậy trung bình của các node biểu quyết có trọng số cao hơn cho các node tin cậy
        avg_trust = np.mean([trust_scores[i] * (1.2 if i in trusted_nodes else 1.0) for i in voting_nodes])
        
        # Xác suất thành công dựa trên điểm tin cậy trung bình, số lượng node và số lượng node tin cậy
        success_prob = min(0.999, avg_trust * 0.98 + 0.02 + 0.1 * len(trusted_nodes) / len(nodes))
        
        # Quyết định kết quả đồng thuận
        consensus_achieved = random.random() < success_prob
        
        # Tính độ trễ
        # Robust BFT có độ trễ cao và phụ thuộc nhiều vào số lượng node và số lượng vòng bỏ phiếu
        latency = self.latency_factor * (1.5 + 0.5 * len(voting_nodes) / len(nodes))
        
        # Tính năng lượng tiêu thụ
        energy = self.energy_factor * (20.0 + 2.0 * len(voting_nodes))
        
        return consensus_achieved, latency, energy

class AdaptiveConsensus:
    """
    Quản lý các giao thức đồng thuận và lựa chọn giao thức phù hợp nhất dựa trên điều kiện mạng,
    giá trị giao dịch và điểm tin cậy của các node.
    """
    def __init__(self):
        self.protocols = {
            "Fast_BFT": FastBFT(),
            "PBFT": PBFT(),
            "Robust_BFT": RobustBFT()
        }
    
    def select_protocol(self, 
                       transaction_value: float,
                       network_congestion: float,
                       trust_scores: List[float],
                       latency_pref: float = 0.5,
                       energy_pref: float = 0.3,
                       security_pref: float = 0.2) -> str:
        """
        Lựa chọn giao thức đồng thuận phù hợp nhất dựa trên các điều kiện.
        
        Args:
            transaction_value: Giá trị giao dịch (0.0-∞)
            network_congestion: Mức độ tắc nghẽn mạng (0.0-1.0)
            trust_scores: Điểm tin cậy của các node (0.0-1.0)
            latency_pref: Hệ số ưu tiên cho độ trễ thấp (0.0-1.0)
            energy_pref: Hệ số ưu tiên cho tiêu thụ năng lượng thấp (0.0-1.0)
            security_pref: Hệ số ưu tiên cho bảo mật cao (0.0-1.0)
            
        Returns:
            str: Tên giao thức đồng thuận được chọn
        """
        # Chuẩn hóa giá trị giao dịch (giá trị càng cao, càng cần bảo mật)
        # Giả sử 100 là ngưỡng giao dịch giá trị cao
        norm_value = min(1.0, transaction_value / 100.0)
        
        # Điểm tin cậy trung bình
        avg_trust = np.mean(trust_scores)
        
        # Đánh giá từng giao thức
        protocol_scores = {}
        
        for name, protocol in self.protocols.items():
            # Điểm cho độ trễ (càng thấp càng tốt)
            latency_score = (1.0 - protocol.latency_factor) * (1.0 - network_congestion)
            
            # Điểm cho tiêu thụ năng lượng (càng thấp càng tốt)
            energy_score = 1.0 - protocol.energy_factor
            
            # Điểm cho bảo mật (càng cao càng tốt)
            # Khi giao dịch có giá trị cao hoặc tin cậy thấp, bảo mật trở nên quan trọng hơn
            security_importance = norm_value + (1.0 - avg_trust)
            security_score = protocol.security_factor * security_importance
            
            # Tổng hợp điểm
            total_score = (latency_score * latency_pref + 
                          energy_score * energy_pref + 
                          security_score * security_pref)
            
            protocol_scores[name] = total_score
        
        # Lựa chọn giao thức có điểm cao nhất
        return max(protocol_scores.items(), key=lambda x: x[1])[0]
    
    def execute_protocol(self, 
                       protocol_name: str, 
                       transaction: Dict[str, Any], 
                       nodes: List[int], 
                       trust_scores: List[float]) -> Tuple[bool, float, float]:
        """
        Thực hiện giao thức đồng thuận đã chọn.
        
        Args:
            protocol_name: Tên giao thức đồng thuận
            transaction: Giao dịch cần được xác nhận
            nodes: Danh sách các node tham gia vào quá trình đồng thuận
            trust_scores: Điểm tin cậy của các node
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        protocol = self.protocols.get(protocol_name)
        if protocol is None:
            raise ValueError(f"Giao thức không hợp lệ: {protocol_name}")
        
        return protocol.execute(transaction, nodes, trust_scores)
    
    def get_protocol_factors(self, protocol_name: str) -> Dict[str, float]:
        """
        Lấy thông tin về các hệ số của giao thức đồng thuận.
        
        Args:
            protocol_name: Tên giao thức đồng thuận
            
        Returns:
            Dict[str, float]: Dictionary chứa các hệ số của giao thức
        """
        protocol = self.protocols.get(protocol_name)
        if protocol is None:
            raise ValueError(f"Giao thức không hợp lệ: {protocol_name}")
        
        return {
            "latency_factor": protocol.latency_factor,
            "energy_factor": protocol.energy_factor,
            "security_factor": protocol.security_factor
        }
    
    def execute_consensus(self, tx: Dict[str, Any], network, protocol_name: str = None) -> Tuple[bool, float, float]:
        """
        Thực hiện giao thức đồng thuận cho một giao dịch.
        
        Args:
            tx: Giao dịch cần xác nhận
            network: Mạng blockchain
            protocol_name: Tên giao thức đồng thuận (nếu None sẽ được tự động lựa chọn)
            
        Returns:
            Tuple[bool, float, float]: (Kết quả đồng thuận, độ trễ, năng lượng tiêu thụ)
        """
        # Nếu không chỉ định giao thức, tự động lựa chọn giao thức tốt nhất
        if protocol_name is None:
            # Lấy thông tin cần thiết từ giao dịch và mạng
            tx_value = tx.get('value', 0.0)
            
            # Ước tính mức độ tắc nghẽn mạng
            dest_shard = tx.get('destination_shard', 0)
            nodes_in_shard = [n for n, data in network.nodes(data=True) 
                             if data.get('shard_id') == dest_shard]
            
            congestion_level = 0.0
            if nodes_in_shard:
                # Tính mức độ tắc nghẽn từ các thuộc tính của node
                congestion_values = [data.get('congestion', 0.0) 
                                    for _, data in network.nodes(data=True) 
                                    if 'congestion' in data and data.get('shard_id') == dest_shard]
                if congestion_values:
                    congestion_level = np.mean(congestion_values)
            
            # Lấy điểm tin cậy của các node
            trust_scores = [data.get('trust_score', 0.5) 
                           for _, data in network.nodes(data=True) 
                           if 'trust_score' in data and data.get('shard_id') == dest_shard]
            
            # Lựa chọn giao thức phù hợp nhất
            protocol_name = self.select_protocol(
                transaction_value=tx_value,
                network_congestion=congestion_level,
                trust_scores=trust_scores
            )
        
        # Lấy các node liên quan đến giao dịch
        # Nếu giao dịch đã chỉ định validator_nodes, sử dụng nó
        # Nếu không, lấy các node trong shard đích
        if 'validator_nodes' in tx and tx['validator_nodes']:
            validator_nodes = tx['validator_nodes']
        else:
            dest_shard = tx.get('destination_shard', 0)
            validator_nodes = [n for n, data in network.nodes(data=True) 
                              if data.get('shard_id') == dest_shard]
        
        # Lấy điểm tin cậy của các validator
        trust_scores = []
        for node in validator_nodes:
            if isinstance(node, int) and network.has_node(node):
                trust_scores.append(network.nodes[node].get('trust_score', 0.5))
            else:
                trust_scores.append(0.5)  # Giá trị mặc định
        
        # Thực hiện giao thức đồng thuận
        result, latency, energy = self.execute_protocol(
            protocol_name=protocol_name,
            transaction=tx,
            nodes=validator_nodes,
            trust_scores=trust_scores
        )
        
        return result, latency, energy 