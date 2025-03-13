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
        # Tối ưu: Cải thiện tính chính xác bằng cách tăng ngưỡng đồng thuận
        if len(validators) < 3:
            return False, 0.0, 0.0

        # Chọn số lượng validator cần thiết (tối thiểu 3, tối đa 30% số validator)
        num_validators_needed = max(3, min(len(validators) // 3, 5))
        
        # Sắp xếp validator theo điểm tin cậy và chọn ra những validator tin cậy nhất
        sorted_validators = sorted(
            [(v, trust_scores.get(v, 0.5)) for v in validators],
            key=lambda x: x[1],
            reverse=True
        )
        
        selected_validators = [v[0] for v in sorted_validators[:num_validators_needed]]
        
        # Mô phỏng quá trình xác minh
        verification_time = time.time()
        
        # Các validator đạt đồng thuận với xác suất cao (dựa vào độ tin cậy)
        avg_trust = sum(trust_scores.get(v, 0.5) for v in selected_validators) / len(selected_validators)
        
        # Cải thiện: Thêm xác suất thành công phụ thuộc vào giá trị giao dịch
        transaction_value = getattr(transaction, 'value', 1.0)
        
        # Giao dịch giá trị cao sẽ có xác suất thành công thấp hơn một chút
        value_factor = 1.0 if transaction_value < 10.0 else (0.98 if transaction_value < 100.0 else 0.95)
        
        # Xác suất thành công dựa vào điểm tin cậy và giá trị giao dịch
        success_probability = min(0.99, avg_trust * value_factor)
        
        # Cải thiện: Lấy mẫu theo phân phối beta thay vì bernoulli để đảm bảo tính ổn định
        success = np.random.beta(success_probability * 10, (1 - success_probability) * 10) > 0.5
        
        # Tính toán năng lượng tiêu thụ và độ trễ
        energy_consumption = self.calculate_energy_consumption(num_validators_needed)
        latency = self.calculate_latency(num_validators_needed, avg_trust)
        
        return success, energy_consumption, latency

    def calculate_energy_consumption(self, num_validators: int) -> float:
        """
        Tính toán năng lượng tiêu thụ cho Fast BFT
        
        Args:
            num_validators: Số lượng validator tham gia
            
        Returns:
            Năng lượng tiêu thụ (đơn vị tương đối)
        """
        # Cải thiện: Năng lượng tiêu thụ phi tuyến tính với số validator
        base_energy = 0.5  # Mức cơ sở thấp cho Fast BFT
        # Cải thiện: Tối ưu hóa năng lượng tiêu thụ với quy mô
        return base_energy + 0.05 * math.log(num_validators + 1)

    def calculate_latency(self, num_validators: int, trust_score: float) -> float:
        """
        Tính toán độ trễ cho Fast BFT
        
        Args:
            num_validators: Số lượng validator tham gia
            trust_score: Điểm tin cậy trung bình
            
        Returns:
            Độ trễ (ms)
        """
        # Cải thiện: Độ trễ phi tuyến tính với số validator và điểm tin cậy
        base_latency = 50.0  # Độ trễ cơ sở cho Fast BFT
        trust_factor = 1.0 - 0.5 * trust_score  # Điểm tin cậy cao giảm độ trễ
        scale_factor = math.log(num_validators + 1) / math.log(4)  # Tăng chậm theo số validator
        
        return base_latency * trust_factor * scale_factor


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
        self.network = None  # Sẽ được thiết lập sau khi khởi tạo
        
        # Các chiến lược đồng thuận
        self.consensus_strategies = {
            "high_trust": FastBFTConsensus(),
            "medium_trust": StandardPBFTConsensus(),
            "low_trust": RobustBFTConsensus()
        }
        
        # Khởi tạo các biến theo dõi hiệu suất
        self.total_transactions = 0
        self.successful_transactions = 0
        self.fast_consensus_usage = 0
        self.standard_consensus_usage = 0
        self.robust_consensus_usage = 0
        self.total_energy_consumption = 0.0
        self.total_latency = 0.0
        self.transaction_values = []
        
        # Biến theo dõi thời gian xử lý
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        self.processing_time_history = []

    def process_cross_shard_transaction(self, 
                                       transaction, 
                                       source_shard_id: int, 
                                       target_shard_id: int,
                                       network) -> Tuple[bool, Dict[str, Any]]:
        """
        Xử lý giao dịch xuyên mảnh sử dụng chiến lược đồng thuận thích ứng
        
        Args:
            transaction: Giao dịch cần xử lý
            source_shard_id: ID của shard nguồn
            target_shard_id: ID của shard đích
            network: Mạng blockchain
            
        Returns:
            success: Kết quả xử lý giao dịch
            stats: Thống kê về quá trình xử lý
        """
        # Đảm bảo tham chiếu đến network
        if self.network is None and network is not None:
            self.network = network
        
        # Đảm bảo các thuộc tính theo dõi hiệu suất được khởi tạo
        if not hasattr(self, 'total_transactions'):
            self.total_transactions = 0
            self.successful_transactions = 0
            self.fast_consensus_usage = 0
            self.standard_consensus_usage = 0
            self.robust_consensus_usage = 0
            self.total_energy_consumption = 0.0
            self.total_latency = 0.0
            self.transaction_values = []
            self.total_processing_time = 0.0
            self.min_processing_time = float('inf')
            self.max_processing_time = 0.0
            self.processing_time_history = []
        
        # Ghi lại thời điểm bắt đầu xử lý
        start_time = time.time()
        
        # Kiểm tra tính hợp lệ của shard ID
        if self.network is None:
            return False, {"error": "Không có tham chiếu đến mạng blockchain"}
        
        if source_shard_id not in self.network.shards:
            return False, {"error": f"Shard nguồn không tồn tại: {source_shard_id}"}
        
        if target_shard_id not in self.network.shards:
            return False, {"error": f"Shard đích không tồn tại: {target_shard_id}"}
            
        # Thu thập danh sách validator từ cả shard nguồn và đích
        source_validators = self.network.shards[source_shard_id].get_validators()
        target_validators = self.network.shards[target_shard_id].get_validators()
        
        # Kết hợp danh sách validator
        all_validators = list(set(source_validators + target_validators))
        
        # Thu thập điểm tin cậy từ trust manager (nếu có)
        trust_scores = {}
        if self.trust_manager:
            try:
                # Lấy điểm tin cậy cho từng validator
                for validator in all_validators:
                    try:
                        trust_scores[validator] = self.trust_manager.get_trust_score(validator)
                    except Exception as e:
                        # Xử lý lỗi khi lấy điểm tin cậy
                        trust_scores[validator] = 0.5  # Điểm tin cậy mặc định
            except Exception as e:
                # Xử lý lỗi từ trust manager
                pass
        
        # Nếu không có đủ validator, thêm một số nút ảo để đảm bảo quá trình đồng thuận
        if len(all_validators) < 3:
            for i in range(3 - len(all_validators)):
                virtual_validator = f"virtual_validator_{i}"
                all_validators.append(virtual_validator)
                trust_scores[virtual_validator] = 0.5  # Điểm tin cậy trung bình cho nút ảo
        
        # Lấy điểm tin cậy trung bình của các validator
        avg_trust_score = sum(trust_scores.get(v, 0.5) for v in all_validators) / len(all_validators)
        
        # Thiết lập giá trị giao dịch mặc định nếu không có
        if not hasattr(transaction, 'value') or transaction.value is None:
            transaction.value = random.uniform(1.0, 100.0)
        
        # Chọn chiến lược đồng thuận dựa trên điểm tin cậy và giá trị giao dịch
        high_value_threshold = 50.0
        
        if transaction.value > high_value_threshold:
            # Giao dịch giá trị cao yêu cầu đảm bảo an toàn hơn
            if avg_trust_score > 0.8:
                consensus_strategy = self.consensus_strategies["medium_trust"]
                self.standard_consensus_usage += 1
            else:
                consensus_strategy = self.consensus_strategies["low_trust"]
                self.robust_consensus_usage += 1
        else:
            # Giao dịch giá trị thấp có thể sử dụng chiến lược nhanh hơn
            if avg_trust_score > 0.85:
                consensus_strategy = self.consensus_strategies["high_trust"]
                self.fast_consensus_usage += 1
            elif avg_trust_score > 0.6:
                consensus_strategy = self.consensus_strategies["medium_trust"]
                self.standard_consensus_usage += 1
            else:
                consensus_strategy = self.consensus_strategies["low_trust"]
                self.robust_consensus_usage += 1
        
        # Xác minh giao dịch sử dụng chiến lược đã chọn
        success, energy_consumption, latency = consensus_strategy.verify_transaction(
            transaction, all_validators, trust_scores
        )
        
        # Nếu xác minh thất bại với chiến lược cao, thử lại với chiến lược an toàn hơn
        if not success and consensus_strategy.name != "Robust BFT":
            # Thử lại với chiến lược mạnh mẽ hơn
            if consensus_strategy.name == "Fast BFT":
                fallback_strategy = self.consensus_strategies["medium_trust"]
                self.standard_consensus_usage += 1
                self.fast_consensus_usage -= 1
            else:
                fallback_strategy = self.consensus_strategies["low_trust"]
                self.robust_consensus_usage += 1
                self.standard_consensus_usage -= 1
                
            # Xác minh lại với chiến lược mạnh mẽ hơn
            fallback_success, fallback_energy, fallback_latency = fallback_strategy.verify_transaction(
                transaction, all_validators, trust_scores
            )
            
            # Cập nhật kết quả
            success = fallback_success
            energy_consumption += fallback_energy  # Cộng dồn năng lượng
            latency += fallback_latency  # Cộng dồn độ trễ (2 lần xác minh)
        
        # Tính thời gian xử lý
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # chuyển đổi sang ms
        
        # Cập nhật thống kê
        self.total_transactions += 1
        if success:
            self.successful_transactions += 1
        
        self.total_energy_consumption += energy_consumption
        self.total_latency += latency
        self.transaction_values.append(transaction.value)
        
        # Cập nhật thống kê thời gian xử lý
        self.total_processing_time += processing_time
        self.min_processing_time = min(self.min_processing_time, processing_time)
        self.max_processing_time = max(self.max_processing_time, processing_time)
        self.processing_time_history.append(processing_time)
        
        # Tạo thống kê
        stats = {
            "success": success,
            "energy_consumption": energy_consumption,
            "latency": latency,
            "trust_score": avg_trust_score,
            "consensus_strategy": consensus_strategy.name,
            "num_validators": len(all_validators),
            "transaction_value": transaction.value,
            "processing_time": processing_time
        }
        
        return success, stats

    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về quá trình đồng thuận
        
        Returns:
            Thống kê về quá trình đồng thuận
        """
        if self.total_transactions == 0:
            return {
                "total_transactions": 0,
                "successful_transactions": 0,
                "consensus_success_rate": 0.0,
                "fast_consensus_percent": 0.0,
                "standard_consensus_percent": 0.0,
                "robust_consensus_percent": 0.0,
                "avg_energy_consumption": 0.0,
                "avg_latency": 0.0,
                "avg_transaction_value": 0.0
            }

        consensus_success_rate = (self.successful_transactions / self.total_transactions) * 100.0

        fast_consensus_percent = (self.fast_consensus_usage / self.total_transactions) * 100.0
        standard_consensus_percent = (self.standard_consensus_usage / self.total_transactions) * 100.0
        robust_consensus_percent = (self.robust_consensus_usage / self.total_transactions) * 100.0

        avg_energy_consumption = self.total_energy_consumption / self.total_transactions
        avg_latency = self.total_latency / self.total_transactions
        avg_transaction_value = sum(self.transaction_values) / len(self.transaction_values)

        # Thống kê thời gian xử lý
        avg_processing_time = self.total_processing_time / self.total_transactions

        stats = {
            "total_transactions": self.total_transactions,
            "successful_transactions": self.successful_transactions,
            "consensus_success_rate": consensus_success_rate,
            "fast_consensus_percent": fast_consensus_percent,
            "standard_consensus_percent": standard_consensus_percent,
            "robust_consensus_percent": robust_consensus_percent,
            "avg_energy_consumption": avg_energy_consumption,
            "avg_latency": avg_latency,
            "avg_transaction_value": avg_transaction_value,
            "avg_processing_time": avg_processing_time,
            "min_processing_time": self.min_processing_time,
            "max_processing_time": self.max_processing_time
        }

        return stats

    def reset_statistics(self) -> None:
        """
        Đặt lại thống kê
        """
        self.total_transactions = 0
        self.successful_transactions = 0
        self.fast_consensus_usage = 0
        self.standard_consensus_usage = 0
        self.robust_consensus_usage = 0
        self.total_energy_consumption = 0.0
        self.total_latency = 0.0
        self.transaction_values = []
        self.total_processing_time = 0.0
        self.min_processing_time = float('inf')
        self.max_processing_time = 0.0
        self.processing_time_history = [] 