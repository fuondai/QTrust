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
        
        # Khởi tạo thuộc tính mới để theo dõi hiệu suất
        self.total_tx_processed = 0
        self.successful_tx_count = 0
        self.avg_processing_time = 0
        self.strategy_success_rates = {
            "high_trust": {"success": 0, "total": 0},
            "medium_trust": {"success": 0, "total": 0},
            "low_trust": {"success": 0, "total": 0}
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
        # Kiểm tra xem Trust Manager có hỗ trợ các API mở rộng không
        if self.trust_manager and hasattr(self.trust_manager, 'get_trust_between_shards'):
            # Lấy điểm tin cậy giữa các shard
            trust_scores = []
            for i in range(len(shards_involved) - 1):
                src_shard = shards_involved[i]
                dst_shard = shards_involved[i+1]
                trust_score = self.trust_manager.get_trust_between_shards(src_shard, dst_shard)
                trust_scores.append(trust_score)
            
            # Lấy điểm tin cậy thấp nhất
            min_trust = min(trust_scores) if trust_scores else 0.5
            
            # Phân tích thành phần quan trọng khác
            # 1. Giá trị giao dịch (cao/thấp)
            is_high_value = transaction_value > 100
            
            # 2. Kiểm tra loại giao dịch
            is_cross_shard = transaction_type == "cross_shard"
            
            # 3. Kiểm tra hiệu suất chiến lược trong quá khứ
            if hasattr(self, 'strategy_success_rates'):
                high_trust_rate = self.strategy_success_rates["high_trust"]["success"] / max(1, self.strategy_success_rates["high_trust"]["total"])
                medium_trust_rate = self.strategy_success_rates["medium_trust"]["success"] / max(1, self.strategy_success_rates["medium_trust"]["total"])
                low_trust_rate = self.strategy_success_rates["low_trust"]["success"] / max(1, self.strategy_success_rates["low_trust"]["total"])
                
                # Nếu một chiến lược có tỷ lệ thành công vượt trội, tăng khả năng chọn nó
                best_rate = max(high_trust_rate, medium_trust_rate, low_trust_rate)
                if best_rate > 0.8 and (best_rate - min(high_trust_rate, medium_trust_rate, low_trust_rate)) > 0.2:
                    if high_trust_rate == best_rate:
                        return "high_trust"
                    elif medium_trust_rate == best_rate:
                        return "medium_trust"
                    else:
                        return "low_trust"
            
            # Kết hợp các yếu tố để quyết định
            if min_trust > 0.8:
                # Độ tin cậy cao
                if is_high_value and is_cross_shard:
                    return "medium_trust"  # Cẩn thận hơn với giao dịch giá trị cao
                else:
                    return "high_trust"
            elif min_trust > 0.5:
                # Độ tin cậy trung bình
                if is_high_value:
                    return "low_trust"  # An toàn hơn với giao dịch giá trị cao
                else:
                    return "medium_trust"
            else:
                # Độ tin cậy thấp
                return "low_trust"
        else:
            # Fallback nếu không có thông tin tin cậy
            if transaction_value > 100:
                return "low_trust"  # Cẩn thận với giao dịch giá trị cao
            else:
                return "medium_trust"

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
        
        # Đảm bảo các thuộc tính theo dõi hiệu suất đã được khởi tạo
        if not hasattr(self, 'total_tx_processed'):
            self.total_tx_processed = 0
        if not hasattr(self, 'successful_tx_count'):
            self.successful_tx_count = 0
        if not hasattr(self, 'strategy_success_rates'):
            self.strategy_success_rates = {
                "high_trust": {"success": 0, "total": 0},
                "medium_trust": {"success": 0, "total": 0},
                "low_trust": {"success": 0, "total": 0}
            }
        
        # Tăng tổng số giao dịch
        self.stats["total_transactions"] += 1
        self.total_tx_processed += 1
        
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
                
        # Đảm bảo có ít nhất 3 validators
        if len(validators) < 3:
            additional_nodes = []
            for shard_id in network.shards:
                if shard_id not in [source_shard_id, target_shard_id]:
                    additional_nodes.extend(network.shards[shard_id].nodes[:2])  # Lấy tối đa 2 node từ mỗi shard khác
                    if len(validators) + len(additional_nodes) >= 3:
                        break
            
            for node_id in additional_nodes:
                if node_id not in validators:
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
        
        # Cập nhật số lượng giao dịch cho chiến lược này
        if strategy_name in self.strategy_success_rates:
            self.strategy_success_rates[strategy_name]["total"] += 1
        
        # Xác minh giao dịch sử dụng chiến lược đã chọn
        success, energy_consumption, latency = strategy.verify_transaction(
            transaction=transaction,
            validators=validators,
            trust_scores=trust_scores
        )
        
        # Tăng tỷ lệ thành công cho giao dịch đặc biệt
        if not success:
            # Thử lại với chiến lược mạnh hơn nếu thất bại
            if strategy_name != "low_trust" and transaction_value > 50:
                print(f"Giao dịch thất bại với chiến lược {strategy_name}, thử lại với chiến lược low_trust")
                strategy = self.consensus_strategies["low_trust"]
                strategy_name = "low_trust"
                success, energy_consumption, latency = strategy.verify_transaction(
                    transaction=transaction,
                    validators=validators,
                    trust_scores=trust_scores
                )
        
        # Cập nhật thống kê
        if success:
            self.stats["successful_transactions"] += 1
            self.successful_tx_count += 1
            
            # Cập nhật tỷ lệ thành công cho chiến lược
            if strategy_name in self.strategy_success_rates:
                self.strategy_success_rates[strategy_name]["success"] += 1
            
            # Gán trạng thái "processed" cho giao dịch nếu thành công
            if hasattr(transaction, 'status'):
                transaction.status = 'processed'
        else:
            # Gán trạng thái "failed" nếu thất bại
            if hasattr(transaction, 'status'):
                transaction.status = 'failed'
        
        self.stats["total_energy"] += energy_consumption
        self.stats["avg_latency"] = ((self.stats["avg_latency"] * (self.stats["total_transactions"] - 1)) 
                                + latency) / self.stats["total_transactions"]
        self.stats["strategy_usage"][strategy_name] += 1
        
        # Tính thời gian xử lý
        processing_time = time.time() - start_time
        if hasattr(self, 'avg_processing_time'):
            self.avg_processing_time = (self.avg_processing_time * (self.total_tx_processed - 1) + processing_time) / self.total_tx_processed
        else:
            self.avg_processing_time = processing_time
        
        # Thống kê cho lần gọi hiện tại
        current_stats = {
            "success": success,
            "energy_consumption": energy_consumption,
            "latency": latency,
            "strategy": strategy_name,
            "processing_time": processing_time,
            "validators_count": len(validators)
        }
        
        return success, current_stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Trả về thống kê về cơ chế đồng thuận xuyên mảnh thích ứng
        
        Returns:
            Dict[str, Any]: Thống kê
        """
        # Đảm bảo các thuộc tính theo dõi hiệu suất đã được khởi tạo
        if not hasattr(self, 'total_tx_processed'):
            self.total_tx_processed = 0
        if not hasattr(self, 'successful_tx_count'):
            self.successful_tx_count = 0
        if not hasattr(self, 'strategy_success_rates'):
            self.strategy_success_rates = {
                "high_trust": {"success": 0, "total": 0},
                "medium_trust": {"success": 0, "total": 0},
                "low_trust": {"success": 0, "total": 0}
            }
            
        # Tính toán các mức độ thành công
        success_rate = self.successful_tx_count / max(1, self.total_tx_processed)
        
        # Tính toán tỷ lệ thành công cho từng chiến lược
        strategy_success_rates = {}
        for strategy, stats in self.strategy_success_rates.items():
            strategy_success_rates[strategy] = stats["success"] / max(1, stats["total"])
        
        # Tạo thống kê chi tiết
        stats = {
            "total_transactions": self.stats["total_transactions"],
            "successful_transactions": self.stats["successful_transactions"],
            "success_rate": success_rate,
            "total_energy": self.stats["total_energy"],
            "avg_latency": self.stats["avg_latency"],
            "strategy_usage": self.stats["strategy_usage"],
            "strategy_success_rates": strategy_success_rates,
            "avg_processing_time": getattr(self, 'avg_processing_time', 0)
        }
        
        # Thêm phân tích về phân bố chiến lược
        total_usage = sum(self.stats["strategy_usage"].values())
        if total_usage > 0:
            strategy_distribution = {}
            for strategy, count in self.stats["strategy_usage"].items():
                strategy_distribution[strategy] = count / total_usage
            stats["strategy_distribution"] = strategy_distribution
        
        # Thêm thông tin về mô hình tin cậy
        if self.trust_manager:
            stats["trust_model"] = {
                "avg_trust_score": sum(self.trust_manager.trust_scores.values()) / max(1, len(self.trust_manager.trust_scores)) if hasattr(self.trust_manager, 'trust_scores') and self.trust_manager.trust_scores else 0,
                "trust_violations": getattr(self.trust_manager, 'trust_violations', 0)
            }
        
        return stats
    
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