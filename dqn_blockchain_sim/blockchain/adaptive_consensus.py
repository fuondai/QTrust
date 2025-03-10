"""
ACSC: Adaptive Cross-Shard Consensus
Module này triển khai cơ chế đồng thuận xuyên mảnh thích ứng dựa trên mức độ tin cậy
và tối ưu hóa năng lượng.
"""

import random
import time
import math
from typing import Dict, List, Any, Set, Tuple, Optional
import numpy as np


class ConsensusStrategy:
    """
    Lớp cơ sở cho các chiến lược đồng thuận
    """
    
    def __init__(self, name: str):
        """
        Khởi tạo chiến lược đồng thuận
        
        Args:
            name: Tên chiến lược
        """
        self.name = name
        
    def verify_transaction(self, 
                        transaction, 
                        validators: List[str], 
                        trust_scores: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Phương thức xác minh giao dịch (sẽ được ghi đè)
        
        Args:
            transaction: Giao dịch cần xác minh
            validators: Danh sách validator
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            success: Kết quả xác minh
            energy_consumption: Năng lượng tiêu thụ
            latency: Độ trễ
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")

    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")
        
    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ của quá trình đồng thuận
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy của shard xử lý
            
        Returns:
            Độ trễ (ms)
        """
        raise NotImplementedError("Phương thức này cần được triển khai bởi lớp con")


class FastBFTConsensus(ConsensusStrategy):
    """
    Chiến lược đồng thuận nhanh cho các shard có độ tin cậy cao
    """
    
    def __init__(self):
        """
        Khởi tạo chiến lược đồng thuận BFT nhanh
        """
        super(FastBFTConsensus, self).__init__("Fast BFT")
        
    def verify_transaction(self, 
                        transaction, 
                        validators: List[str], 
                        trust_scores: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Xác minh giao dịch bằng cách chọn một số lượng nhỏ validator tin cậy
        
        Args:
            transaction: Giao dịch cần xác minh
            validators: Danh sách validator
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            success: Kết quả xác minh
            energy_consumption: Năng lượng tiêu thụ
            latency: Độ trễ
        """
        # Sắp xếp validators theo điểm tin cậy
        sorted_validators = sorted(
            [(v, trust_scores.get(v, 0.5)) for v in validators],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Chỉ sử dụng 33% validators có điểm tin cậy cao nhất
        num_validators = max(3, int(len(validators) * 0.33))
        selected_validators = [v for v, _ in sorted_validators[:num_validators]]
        
        # Tính xác suất thành công
        avg_trust = sum(trust_scores.get(v, 0.5) for v in selected_validators) / len(selected_validators)
        success_prob = min(0.99, avg_trust * 0.95 + 0.05)  # Weighted with base 5% chance
        
        # Mô phỏng quá trình xác minh - nếu tin cậy cao, xác suất thành công cao
        success = random.random() < success_prob
        
        # Năng lượng tiêu thụ thấp do số lượng validator ít
        energy_per_validator = 0.8  # 80% so với BFT tiêu chuẩn
        energy_consumption = energy_per_validator * len(selected_validators)
        
        # Độ trễ thấp do ít validator
        base_latency = 100  # Độ trễ cơ bản (ms)
        validator_factor = math.log2(max(2, len(selected_validators)))  # Tăng logarit theo số validator
        latency = base_latency * validator_factor
        
        return success, energy_consumption, latency

    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ cho chiến lược nhanh
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        # Tiêu thụ năng lượng thấp
        base_energy = 5.0
        return base_energy * num_validators
        
    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ cho chiến lược nhanh
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy của shard xử lý
            
        Returns:
            Độ trễ (ms)
        """
        # Độ trễ thấp
        base_latency = 100.0
        return base_latency * (1 + (num_validators * 0.05))


class StandardPBFTConsensus(ConsensusStrategy):
    """
    Chiến lược đồng thuận PBFT tiêu chuẩn cho các shard có độ tin cậy trung bình
    """
    
    def __init__(self):
        """
        Khởi tạo chiến lược đồng thuận PBFT tiêu chuẩn
        """
        super(StandardPBFTConsensus, self).__init__("Standard PBFT")
        
    def verify_transaction(self, 
                        transaction, 
                        validators: List[str], 
                        trust_scores: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Xác minh giao dịch bằng thuật toán PBFT tiêu chuẩn
        
        Args:
            transaction: Giao dịch cần xác minh
            validators: Danh sách validator
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            success: Kết quả xác minh
            energy_consumption: Năng lượng tiêu thụ
            latency: Độ trễ
        """
        # Sử dụng 66% validators
        num_validators = max(4, int(len(validators) * 0.66))
        selected_validators = random.sample(validators, min(num_validators, len(validators)))
        
        # Tính điểm tin cậy trung bình
        avg_trust = sum(trust_scores.get(v, 0.5) for v in selected_validators) / len(selected_validators)
        
        # Mô phỏng quá trình PBFT tìm đồng thuận
        quorum_size = 2 * len(selected_validators) // 3 + 1  # Quá bán +1
        
        # Mỗi validator có cơ hội xác minh thành công dựa trên điểm tin cậy của họ
        successful_validations = 0
        for validator in selected_validators:
            trust = trust_scores.get(validator, 0.5)
            if random.random() < (trust * 0.9 + 0.1):  # Tỷ lệ thành công dựa vào tin cậy
                successful_validations += 1
                
        # Đạt đồng thuận nếu có đủ số lượng validator thành công
        success = successful_validations >= quorum_size
        
        # Năng lượng tiêu thụ
        energy_per_validator = 1.0  # Năng lượng chuẩn
        energy_consumption = energy_per_validator * len(selected_validators)
        
        # Độ trễ - PBFT có 3 giai đoạn
        base_latency = 200  # Độ trễ cơ bản (ms)
        phases = 3  # Pre-prepare, prepare, commit
        validator_factor = math.log2(max(2, len(selected_validators)))
        latency = base_latency * phases * validator_factor
        
        return success, energy_consumption, latency

    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ cho chiến lược tiêu chuẩn
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        # Tiêu thụ năng lượng trung bình
        base_energy = 10.0
        return base_energy * num_validators * 1.5  # PBFT tốn nhiều năng lượng hơn FastBFT
        
    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ cho chiến lược tiêu chuẩn
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy của shard xử lý
            
        Returns:
            Độ trễ (ms)
        """
        # Độ trễ trung bình
        base_latency = 200.0
        # PBFT có độ phức tạp O(n²)
        return base_latency * (1 + (num_validators ** 2) * 0.005)


class RobustBFTConsensus(ConsensusStrategy):
    """
    Chiến lược đồng thuận BFT mạnh mẽ cho các shard có độ tin cậy thấp hoặc giao dịch giá trị cao
    """
    
    def __init__(self):
        """
        Khởi tạo chiến lược đồng thuận BFT mạnh mẽ
        """
        super(RobustBFTConsensus, self).__init__("Robust BFT")
        
    def verify_transaction(self, 
                        transaction, 
                        validators: List[str], 
                        trust_scores: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Xác minh giao dịch với yêu cầu cao hơn và kiểm tra bổ sung
        
        Args:
            transaction: Giao dịch cần xác minh
            validators: Danh sách validator
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            success: Kết quả xác minh
            energy_consumption: Năng lượng tiêu thụ
            latency: Độ trễ
        """
        # Sử dụng 90% validators và yêu cầu đồng thuận cao
        num_validators = max(5, int(len(validators) * 0.9))
        selected_validators = random.sample(validators, min(num_validators, len(validators)))
        
        # Tính điểm tin cậy trung bình
        avg_trust = sum(trust_scores.get(v, 0.5) for v in selected_validators) / len(selected_validators)
        
        # Quorum cao - 75% thay vì 66%
        quorum_size = 3 * len(selected_validators) // 4 + 1
        
        # Khởi tạo đếm xác thực
        successful_validations = 0
        
        # Mỗi validator thực hiện kiểm tra bổ sung
        for validator in selected_validators:
            trust = trust_scores.get(validator, 0.5)
            
            # Mô phỏng kiểm tra kỹ lưỡng hơn
            basic_check = random.random() < (trust * 0.9 + 0.1)
            extra_check = random.random() < (trust * 0.95 + 0.05)  # Kiểm tra bổ sung
            
            if basic_check and extra_check:
                successful_validations += 1
                
        # Đạt đồng thuận nếu có đủ số lượng validator thành công
        success = successful_validations >= quorum_size
        
        # Năng lượng tiêu thụ cao hơn do kiểm tra bổ sung
        energy_per_validator = 1.5  # 150% so với tiêu chuẩn
        energy_consumption = energy_per_validator * len(selected_validators)
        
        # Độ trễ cao hơn do kiểm tra bổ sung và quorum cao hơn
        base_latency = 300  # Độ trễ cơ bản (ms)
        phases = 4  # Thêm giai đoạn xác minh
        validator_factor = math.log2(max(2, len(selected_validators)))
        latency = base_latency * phases * validator_factor
        
        return success, energy_consumption, latency

    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ cho chiến lược mạnh mẽ
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        # Tiêu thụ năng lượng cao
        base_energy = 15.0
        return base_energy * num_validators * 2.5  # Thuật toán mạnh mẽ tốn nhiều năng lượng hơn
        
    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ cho chiến lược mạnh mẽ
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy của shard xử lý
            
        Returns:
            Độ trễ (ms)
        """
        # Độ trễ cao
        base_latency = 350.0
        # Thuật toán mạnh mẽ có độ phức tạp cao hơn
        return base_latency * (1 + (num_validators ** 2) * 0.01)


class AdaptiveCrossShardConsensus:
    """
    Cơ chế đồng thuận xuyên mảnh thích ứng dựa trên mức độ tin cậy và tối ưu hóa năng lượng
    """
    
    def __init__(self, trust_manager=None):
        """
        Khởi tạo cơ chế đồng thuận xuyên mảnh thích ứng
        
        Args:
            trust_manager: Tham chiếu đến trình quản lý tin cậy (nếu có)
        """
        self.trust_manager = trust_manager
        
        # Các chiến lược đồng thuận
        self.consensus_strategies = {
            "high_trust": FastBFTConsensus(),
            "medium_trust": StandardPBFTConsensus(),
            "low_trust": RobustBFTConsensus()
        }
        
        # Thống kê
        self.stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "total_energy": 0.0,
            "avg_latency": 0.0,
            "strategy_usage": {
                "high_trust": 0,
                "medium_trust": 0,
                "low_trust": 0
            }
        }
        
    def select_consensus_strategy(self, 
                                shards_involved: List[int], 
                                transaction_value: float,
                                network,
                                transaction_type: str = "standard") -> str:
        """
        Chọn chiến lược đồng thuận dựa trên nhiều yếu tố
        
        Args:
            shards_involved: Danh sách các shard liên quan
            transaction_value: Giá trị giao dịch
            network: Tham chiếu đến mạng blockchain
            transaction_type: Loại giao dịch
            
        Returns:
            Tên chiến lược được chọn
        """
        # Tính điểm tin cậy trung bình của các shard
        avg_trust = 0.0
        
        if len(shards_involved) > 0:
            trust_sum = 0.0
            node_count = 0
            
            # Tính điểm tin cậy trung bình cho tất cả các nút trong tất cả các shard liên quan
            for shard_id in shards_involved:
                if shard_id in network.shards:
                    shard = network.shards[shard_id]
                    for node_id in shard.nodes:
                        if self.trust_manager:
                            trust_sum += self.trust_manager.get_trust_score(node_id)
                        else:
                            trust_sum += 0.7  # Giá trị mặc định nếu không có trust_manager
                        node_count += 1
                        
            if node_count > 0:
                avg_trust = trust_sum / node_count
                
        # Điều chỉnh dựa trên giá trị giao dịch
        # Giao dịch giá trị cao -> bảo mật cao hơn
        value_factor = min(1.0, transaction_value / 1000)  # Chuẩn hóa giá trị
        
        # Điều chỉnh dựa trên loại giao dịch
        type_factor = 0.0
        if transaction_type == "high_security":
            type_factor = 0.3  # Giảm điểm tin cậy, yêu cầu bảo mật cao hơn
        elif transaction_type == "fast":
            type_factor = -0.2  # Tăng điểm tin cậy, tốc độ trên hết
            
        # Tính toán điểm cuối cùng
        final_trust = max(0.0, min(1.0, avg_trust - value_factor * 0.3 - type_factor))
        
        # Chọn chiến lược dựa trên điểm tin cậy
        if final_trust >= 0.7:
            return "high_trust"
        elif final_trust >= 0.4:
            return "medium_trust"
        else:
            return "low_trust"
            
    def process_cross_shard_transaction(self, 
                                     transaction, 
                                     source_shard_id: int, 
                                     target_shard_id: int,
                                     network) -> Tuple[bool, Dict[str, Any]]:
        """
        Xử lý giao dịch xuyên mảnh bằng cách chọn và áp dụng chiến lược đồng thuận thích hợp
        
        Args:
            transaction: Giao dịch cần xử lý
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            network: Tham chiếu đến mạng blockchain
            
        Returns:
            success: Kết quả xử lý
            stats: Thống kê về quá trình xử lý
        """
        # Lấy thông tin về giao dịch
        tx_value = transaction.amount if hasattr(transaction, 'amount') else 50.0
        tx_type = transaction.data.get('type', 'standard') if hasattr(transaction, 'data') else 'standard'
        
        # Chọn chiến lược đồng thuận
        strategy_name = self.select_consensus_strategy(
            [source_shard_id, target_shard_id],
            tx_value,
            network,
            tx_type
        )
        
        # Lấy chiến lược đã chọn
        consensus_strategy = self.consensus_strategies[strategy_name]
        
        # Thu thập danh sách validator từ cả hai shard
        validators = []
        
        if source_shard_id in network.shards:
            validators.extend(list(network.shards[source_shard_id].nodes))
            
        if target_shard_id in network.shards and target_shard_id != source_shard_id:
            validators.extend(list(network.shards[target_shard_id].nodes))
            
        # Lấy điểm tin cậy
        trust_scores = {}
        if self.trust_manager:
            for validator in validators:
                trust_scores[validator] = self.trust_manager.get_trust_score(validator)
                
        # Xác minh giao dịch
        start_time = time.time()
        success, energy_consumption, latency = consensus_strategy.verify_transaction(
            transaction, validators, trust_scores
        )
        
        # Cập nhật thống kê
        self.stats["total_transactions"] += 1
        if success:
            self.stats["successful_transactions"] += 1
            
        self.stats["total_energy"] += energy_consumption
        self.stats["avg_latency"] = ((self.stats["avg_latency"] * (self.stats["total_transactions"] - 1)) 
                                + latency) / self.stats["total_transactions"]
        self.stats["strategy_usage"][strategy_name] += 1
        
        # Thống kê cho lần gọi hiện tại
        current_stats = {
            'success': success,
            'energy_consumption': energy_consumption,
            'latency': latency,
            'strategy': strategy_name,
            'processing_time': time.time() - start_time
        }
        
        return success, current_stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hoạt động của cơ chế đồng thuận
        
        Returns:
            Từ điển chứa các thống kê
        """
        # Tính tỉ lệ thành công
        success_rate = 0.0
        if self.stats["total_transactions"] > 0:
            success_rate = self.stats["successful_transactions"] / self.stats["total_transactions"]
            
        # Tính tỷ lệ sử dụng các chiến lược
        strategy_percentages = {}
        if self.stats["total_transactions"] > 0:
            for strategy, count in self.stats["strategy_usage"].items():
                strategy_percentages[strategy] = count / self.stats["total_transactions"]
                
        # Tổng hợp thống kê
        return {
            'total_transactions': self.stats["total_transactions"],
            'successful_transactions': self.stats["successful_transactions"],
            'success_rate': success_rate,
            'total_energy': self.stats["total_energy"],
            'avg_energy_per_tx': (self.stats["total_energy"] / max(1, self.stats["total_transactions"])),
            'avg_latency': self.stats["avg_latency"],
            'strategy_usage': self.stats["strategy_usage"],
            'strategy_percentages': strategy_percentages
        }
    
    def reset_statistics(self) -> None:
        """
        Đặt lại thống kê
        """
        self.stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "total_energy": 0.0,
            "avg_latency": 0.0,
            "strategy_usage": {
                "high_trust": 0,
                "medium_trust": 0,
                "low_trust": 0
            }
        } 