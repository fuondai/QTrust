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
    Chiến lược đồng thuận mạnh mẽ sử dụng 3f+1 validator
    Đảm bảo an toàn cao nhất, nhưng có độ trễ và tiêu thụ năng lượng cao
    """
    
    def __init__(self):
        """
        Khởi tạo chiến lược đồng thuận mạnh mẽ
        """
        super().__init__("RobustBFTConsensus")
        
    def verify_transaction(self, 
                        transaction, 
                        validators: List[str], 
                        trust_scores: Dict[str, float]) -> Tuple[bool, float, float]:
        """
        Xác minh giao dịch sử dụng chiến lược BFT mạnh mẽ
        
        Args:
            transaction: Giao dịch cần xác minh
            validators: Danh sách validator
            trust_scores: Điểm tin cậy của các validator
            
        Returns:
            success: Kết quả xác minh
            energy_consumption: Năng lượng tiêu thụ
            latency: Độ trễ
        """
        # Số lượng validator tối thiểu cần thiết để đạt đồng thuận
        min_validators = 3 * self._max_faulty_nodes(len(validators)) + 1
        
        # Nếu không đủ validator, trả về False
        if len(validators) < min_validators:
            return False, 0.0, 0.0
            
        # Xác minh giao dịch
        valid_votes = 0
        
        # Giả lập quá trình xác minh
        for validator in validators:
            # Xét điểm tin cậy của validator
            trust = trust_scores.get(validator, 0.5)
            
            # Validator có điểm tin cậy thấp có thể từ chối đúng giao dịch
            if random.random() < 0.8 * trust:  # 80% * trust_score
                valid_votes += 1
                
        # Kiểm tra đồng thuận: cần ít nhất 2/3 validator đồng ý
        consensus_threshold = len(validators) * 2 / 3
        success = valid_votes >= consensus_threshold
        
        # Tính năng lượng tiêu thụ (đơn vị tương đối)
        energy_consumption = self.calculate_energy_consumption(len(validators))
        
        # Tính độ trễ (ms)
        avg_trust = sum(trust_scores.values()) / len(trust_scores) if trust_scores else 0.5
        latency = self.calculate_latency(len(validators), avg_trust)
        
        # Trả về kết quả
        return success, energy_consumption, latency
    
    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        # Công thức ước tính: mỗi validator tiêu thụ 5 đơn vị năng lượng
        return num_validators * 5
        
    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy trung bình
            
        Returns:
            Độ trễ (ms)
        """
        # Công thức ước tính: độ trễ tỷ lệ với số lượng validator và giảm theo điểm tin cậy
        base_latency = 200  # ms
        return base_latency * (1 + 0.1 * num_validators) * (1 - 0.2 * trust_score)
        
    def _max_faulty_nodes(self, total_nodes: int) -> int:
        """
        Tính toán số lượng node lỗi tối đa có thể chấp nhận
        
        Args:
            total_nodes: Tổng số node
            
        Returns:
            Số lượng node lỗi tối đa
        """
        return (total_nodes - 1) // 3


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
        Lựa chọn chiến lược đồng thuận phù hợp dựa trên ngữ cảnh
        
        Args:
            shards_involved: Danh sách các shard liên quan
            transaction_value: Giá trị giao dịch
            network: Mạng blockchain
            transaction_type: Loại giao dịch
            
        Returns:
            Tên chiến lược đồng thuận được chọn
        """
        # Yếu tố 1: Đánh giá điểm tin cậy trung bình của các shard liên quan
        avg_trust_score = 0.0
        total_trust_score = 0.0
        total_nodes = 0
        
        for shard_id in shards_involved:
            if shard_id in network.shards:
                shard = network.shards[shard_id]
                for node_id in shard.nodes:
                    # Lấy điểm tin cậy nếu trust_manager được cung cấp
                    trust_score = self.trust_manager.get_trust_score(node_id) if self.trust_manager else 0.5
                    total_trust_score += trust_score
                    total_nodes += 1
        
        # Tính điểm tin cậy trung bình
        avg_trust_score = total_trust_score / total_nodes if total_nodes > 0 else 0.5
        
        # Yếu tố 2: Đánh giá giá trị giao dịch
        # Ngưỡng giá trị: thấp < 10, trung bình 10-100, cao > 100
        value_level = 0
        if transaction_value < 10:
            value_level = 0  # thấp
        elif transaction_value < 100:
            value_level = 1  # trung bình
        else:
            value_level = 2  # cao
        
        # Yếu tố 3: Loại giao dịch
        # Giao dịch xuyên shard cần độ tin cậy cao hơn
        is_cross_shard = transaction_type == "cross_shard"
        
        # Xác suất cho mỗi chiến lược
        high_trust_prob = 0.0
        medium_trust_prob = 0.0
        low_trust_prob = 0.0
        
        # Tính xác suất dựa trên điểm tin cậy
        if avg_trust_score > 0.8:
            high_trust_prob = 0.7
            medium_trust_prob = 0.2
            low_trust_prob = 0.1
        elif avg_trust_score > 0.5:
            high_trust_prob = 0.3
            medium_trust_prob = 0.6
            low_trust_prob = 0.1
        else:
            high_trust_prob = 0.1
            medium_trust_prob = 0.3
            low_trust_prob = 0.6
        
        # Điều chỉnh dựa trên giá trị giao dịch
        if value_level == 2:  # Giá trị cao
            low_trust_prob += 0.2
            medium_trust_prob += 0.1
            high_trust_prob -= 0.3
        elif value_level == 1:  # Giá trị trung bình
            medium_trust_prob += 0.2
            high_trust_prob -= 0.1
            low_trust_prob -= 0.1
        else:  # Giá trị thấp
            high_trust_prob += 0.2
            medium_trust_prob -= 0.1
            low_trust_prob -= 0.1
        
        # Chuẩn hóa xác suất
        total_prob = high_trust_prob + medium_trust_prob + low_trust_prob
        high_trust_prob /= total_prob
        medium_trust_prob /= total_prob
        low_trust_prob /= total_prob
        
        # Điều chỉnh nếu là giao dịch xuyên shard
        if is_cross_shard:
            # Tăng xác suất sử dụng low_trust cho giao dịch xuyên shard
            adjustment = 0.2
            low_trust_prob += adjustment
            high_trust_prob -= adjustment / 2
            medium_trust_prob -= adjustment / 2
            
            # Chuẩn hóa lại
            total_prob = high_trust_prob + medium_trust_prob + low_trust_prob
            high_trust_prob = max(0, high_trust_prob / total_prob)
            medium_trust_prob = max(0, medium_trust_prob / total_prob)
            low_trust_prob = max(0, low_trust_prob / total_prob)
        
        # Print debug info
        print(f"Trust scores - High: {high_trust_prob:.2f}, Medium: {medium_trust_prob:.2f}, Low: {low_trust_prob:.2f}")
        
        # Đảm bảo có sự đa dạng trong việc lựa chọn chiến lược
        rand_val = random.random()
        
        # Lựa chọn chiến lược theo xác suất
        if rand_val < high_trust_prob:
            return "high_trust"
        elif rand_val < high_trust_prob + medium_trust_prob:
            return "medium_trust"
        else:
            return "low_trust"
            
    def process_cross_shard_transaction(self, 
                                     transaction, 
                                     source_shard_id: int, 
                                     target_shard_id: int,
                                     network) -> Tuple[bool, Dict[str, Any]]:
        """
        Xử lý giao dịch xuyên mảnh sử dụng chiến lược đồng thuận thích ứng
        
        Args:
            transaction: Giao dịch cần xử lý
            source_shard_id: ID mảnh nguồn
            target_shard_id: ID mảnh đích
            network: Đối tượng mạng blockchain
            
        Returns:
            success: Kết quả xử lý
            stats: Thống kê về quá trình xử lý
        """
        start_time = time.time()
        
        # Tăng tổng số giao dịch
        self.stats["total_transactions"] += 1
        
        # Lấy danh sách validators và điểm tin cậy
        validators = []
        trust_scores = {}
        
        for node_id in network.shards[source_shard_id].nodes:
            validators.append(node_id)
            trust_scores[node_id] = self.trust_manager.get_trust_score(node_id) if self.trust_manager else 0.5
            
        for node_id in network.shards[target_shard_id].nodes:
            if node_id not in validators:  # Tránh trùng lặp nếu node thuộc cả hai mảnh
                validators.append(node_id)
                trust_scores[node_id] = self.trust_manager.get_trust_score(node_id) if self.trust_manager else 0.5
        
        # Lựa chọn chiến lược đồng thuận dựa trên các yếu tố
        shards_involved = [source_shard_id, target_shard_id]
        transaction_value = transaction.value if hasattr(transaction, 'value') else 0
        
        strategy_name = self.select_consensus_strategy(
            shards_involved=shards_involved,
            transaction_value=transaction_value,
            network=network,
            transaction_type="cross_shard"
        )
        
        # Lấy chiến lược tương ứng
        strategy = None
        if strategy_name == "high_trust":
            strategy = self.consensus_strategies["high_trust"]
        elif strategy_name == "medium_trust":
            strategy = self.consensus_strategies["medium_trust"]
        elif strategy_name == "low_trust":
            strategy = self.consensus_strategies["low_trust"]
        
        # Xác minh giao dịch sử dụng chiến lược đã chọn
        success, energy_consumption, latency = strategy.verify_transaction(
            transaction=transaction,
            validators=validators,
            trust_scores=trust_scores
        )
        
        # Cập nhật thống kê
        if success:
            self.stats["successful_transactions"] += 1
            
            # Gán trạng thái "processed" cho giao dịch nếu thành công
            if hasattr(transaction, 'status'):
                transaction.status = 'processed'
        
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
        
        # Log các giá trị quan trọng để debug
        print(f"ACSC Stats: Total={self.stats['total_transactions']}, Success={self.stats['successful_transactions']}, Rate={success_rate}")
        print(f"Strategy Usage: {self.stats['strategy_usage']}")
            
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