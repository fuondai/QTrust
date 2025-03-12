"""
Module quản lý tin cậy và lựa chọn nút dựa trên độ tin cậy (TDCM)
"""

import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set
import random
from collections import defaultdict, deque

from dqn_blockchain_sim.configs.simulation_config import TDCM_CONFIG


class TrustScoreCalculator:
    """
    Lớp tính toán điểm tin cậy cho các nút
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo bộ tính toán điểm tin cậy
        
        Args:
            config: Cấu hình TDCM, sử dụng mặc định nếu không cung cấp
        """
        self.config = config if config is not None else TDCM_CONFIG
        self.trust_factors = self.config["trust_factors"]
        
    def calculate_trust_score(self, node_metrics: Dict[str, Any]) -> float:
        """
        Tính toán điểm tin cậy cho một nút dựa trên các chỉ số
        
        Args:
            node_metrics: Từ điển chứa các chỉ số của nút
            
        Returns:
            Điểm tin cậy (0-1)
        """
        if not node_metrics:
            return 0.0
            
        # Sử dụng thuật toán đánh giá tin cậy đa yếu tố
        score = 0.0
        
        # 1. Thời gian hoạt động
        if "uptime" in node_metrics:
            uptime_score = min(1.0, node_metrics["uptime"])
            score += self.trust_factors["uptime"] * uptime_score
            
        # 2. Hiệu suất
        if "performance" in node_metrics:
            perf_score = min(1.0, node_metrics["performance"])
            score += self.trust_factors["performance"] * perf_score
            
        # 3. Các sự cố bảo mật
        if "security_incidents" in node_metrics:
            # Số sự cố cao -> điểm thấp
            incident_count = node_metrics["security_incidents"]
            security_score = max(0.0, 1.0 - min(1.0, incident_count / 10.0))
            score += self.trust_factors["security_incidents"] * security_score
            
        # 4. Hiệu quả năng lượng
        if "energy_efficiency" in node_metrics:
            energy_score = min(1.0, node_metrics["energy_efficiency"])
            score += self.trust_factors["energy_efficiency"] * energy_score
            
        # Đảm bảo điểm tin cậy nằm trong khoảng [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score


class TrustHistory:
    """
    Lớp lưu trữ và quản lý lịch sử điểm tin cậy
    """
    
    def __init__(self, max_history: int = 100, decay_rate: float = 0.01):
        """
        Khởi tạo lịch sử tin cậy
        
        Args:
            max_history: Số lượng điểm tin cậy tối đa được lưu trữ
            decay_rate: Tốc độ suy giảm cho các điểm tin cậy cũ
        """
        self.max_history = max_history
        self.decay_rate = decay_rate
        self.trust_history = defaultdict(lambda: deque(maxlen=max_history))
        self.timestamps = defaultdict(lambda: deque(maxlen=max_history))
        
    def add_score(self, node_id: str, score: float) -> None:
        """
        Thêm điểm tin cậy mới vào lịch sử
        
        Args:
            node_id: ID của nút
            score: Điểm tin cậy mới
        """
        self.trust_history[node_id].append(score)
        self.timestamps[node_id].append(time.time())
        
    def get_weighted_score(self, node_id: str, history_weight: float = 0.7) -> float:
        """
        Lấy điểm tin cậy có trọng số dựa trên lịch sử
        
        Args:
            node_id: ID của nút
            history_weight: Trọng số cho lịch sử (so với điểm mới nhất)
            
        Returns:
            Điểm tin cậy trung bình có trọng số
        """
        if node_id not in self.trust_history or not self.trust_history[node_id]:
            return 0.0
            
        # Lấy điểm tin cậy gần nhất
        latest_score = self.trust_history[node_id][-1]
        
        if len(self.trust_history[node_id]) == 1:
            return latest_score
            
        # Tính toán điểm lịch sử có trọng số
        history = list(self.trust_history[node_id])[:-1]  # Không tính điểm mới nhất
        timestamps = list(self.timestamps[node_id])[:-1]
        
        if not history:
            return latest_score
            
        # Tính toán trọng số dựa trên thời gian (điểm cũ có trọng số thấp hơn)
        current_time = time.time()
        time_weights = [np.exp(-self.decay_rate * (current_time - ts)) for ts in timestamps]
        
        # Chuẩn hóa trọng số thời gian
        sum_weights = sum(time_weights)
        if sum_weights > 0:
            time_weights = [w / sum_weights for w in time_weights]
        else:
            time_weights = [1.0 / len(history)] * len(history)
            
        # Tính điểm trung bình có trọng số
        historical_score = sum(s * w for s, w in zip(history, time_weights))
        
        # Kết hợp điểm lịch sử và điểm mới nhất
        return (history_weight * historical_score) + ((1 - history_weight) * latest_score)


class TrustManager:
    """
    Lớp quản lý tin cậy và lựa chọn nút
    """
    
    def __init__(self, network=None, num_shards=None, config: Dict[str, Any] = None):
        """
        Khởi tạo trình quản lý tin cậy
        
        Args:
            network: Đối tượng mạng blockchain
            num_shards: Số lượng shard trong mạng
            config: Cấu hình TDCM, sử dụng mặc định nếu không cung cấp
        """
        self.config = config if config is not None else TDCM_CONFIG
        self.network = network
        self.num_shards = num_shards if num_shards is not None else 8
        
        self.trust_calculator = TrustScoreCalculator(self.config)
        self.trust_history = TrustHistory(
            max_history=100,
            decay_rate=self.config["trust_decay_rate"]
        )
        
        self.nodes = {}  # Thông tin về các nút (node_id -> node_info)
        self.trust_scores = {}  # Điểm tin cậy hiện tại (node_id -> score)
        self.last_update_time = {}  # Thời gian cập nhật cuối cùng (node_id -> timestamp)
        
        # Khởi tạo điểm tin cậy cho các shard
        if self.num_shards:
            self.trust_scores = {i: 0.5 for i in range(self.num_shards)}
            self.reputation_history = {}
        
        # Lưu trữ thông tin lựa chọn
        self.selection_history = []
        
    def reset_statistics(self):
        """
        Đặt lại thống kê về tin cậy
        """
        self.trust_scores = {i: 0.5 for i in range(self.num_shards)}
        self.reputation_history = {}
        
    def add_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """
        Thêm một nút vào hệ thống
        
        Args:
            node_id: ID của nút
            node_info: Thông tin về nút
        """
        self.nodes[node_id] = node_info
        
        # Tính toán điểm tin cậy ban đầu
        initial_metrics = {
            "uptime": node_info.get("uptime", 1.0),
            "performance": node_info.get("compute_power", 0.5),
            "security_incidents": 0,
            "energy_efficiency": node_info.get("energy_efficiency", 0.7)
        }
        
        initial_score = self.trust_calculator.calculate_trust_score(initial_metrics)
        self.trust_scores[node_id] = initial_score
        self.trust_history.add_score(node_id, initial_score)
        self.last_update_time[node_id] = time.time()
        
    def remove_node(self, node_id: str) -> bool:
        """
        Xóa một nút khỏi hệ thống
        
        Args:
            node_id: ID của nút cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.trust_scores:
                del self.trust_scores[node_id]
            if node_id in self.last_update_time:
                del self.last_update_time[node_id]
            return True
        return False
        
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]) -> float:
        """
        Cập nhật các chỉ số cho một nút và tính toán lại điểm tin cậy
        
        Args:
            node_id: ID của nút
            metrics: Các chỉ số mới
            
        Returns:
            Điểm tin cậy mới
        """
        if node_id not in self.nodes:
            return 0.0
            
        # Cập nhật thông tin nút với các chỉ số mới
        self.nodes[node_id].update(metrics)
        
        # Tạo từ điển chỉ số cho việc tính toán điểm tin cậy
        trust_metrics = {
            "uptime": metrics.get("uptime", self.nodes[node_id].get("uptime", 1.0)),
            "performance": metrics.get("performance", self.nodes[node_id].get("compute_power", 0.5)),
            "security_incidents": metrics.get("security_incidents", 0),
            "energy_efficiency": metrics.get("energy_efficiency", self.nodes[node_id].get("energy_efficiency", 0.7))
        }
        
        # Tính toán điểm tin cậy mới
        new_score = self.trust_calculator.calculate_trust_score(trust_metrics)
        
        # Cập nhật điểm tin cậy và lịch sử
        self.trust_scores[node_id] = new_score
        self.trust_history.add_score(node_id, new_score)
        self.last_update_time[node_id] = time.time()
        
        return new_score
    
    def batch_update_trust_scores(self, metrics_batch: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Cập nhật hàng loạt điểm tin cậy cho nhiều nút
        
        Args:
            metrics_batch: Từ điển ánh xạ từ node_id đến từ điển chỉ số
            
        Returns:
            Từ điển ánh xạ từ node_id đến điểm tin cậy mới
        """
        updated_scores = {}
        
        for node_id, metrics in metrics_batch.items():
            if node_id in self.nodes:
                updated_scores[node_id] = self.update_node_metrics(node_id, metrics)
                
        return updated_scores
    
    def get_trust_score(self, node_id: str) -> float:
        """
        Lấy điểm tin cậy hiện tại của một nút
        
        Args:
            node_id: ID của nút
            
        Returns:
            Điểm tin cậy, hoặc 0.0 nếu không tìm thấy
        """
        if node_id not in self.trust_scores:
            return 0.0
            
        # Lấy điểm tin cậy có trọng số dựa trên lịch sử
        return self.trust_history.get_weighted_score(
            node_id, 
            self.config["trust_history_weight"]
        )
    
    def get_all_trust_scores(self) -> Dict[str, float]:
        """
        Lấy điểm tin cậy của tất cả các nút
        
        Returns:
            Từ điển ánh xạ từ node_id đến điểm tin cậy
        """
        return {
            node_id: self.get_trust_score(node_id)
            for node_id in self.nodes.keys()
        }
    
    def select_nodes(self, 
                    num_nodes: int, 
                    min_trust: float = None,
                    required_attributes: Dict[str, Any] = None) -> List[str]:
        """
        Lựa chọn các nút dựa trên điểm tin cậy và các thuộc tính bắt buộc
        
        Args:
            num_nodes: Số lượng nút cần chọn
            min_trust: Ngưỡng tin cậy tối thiểu, sử dụng giá trị từ cấu hình nếu không cung cấp
            required_attributes: Từ điển các thuộc tính yêu cầu (tùy chọn)
            
        Returns:
            Danh sách ID của các nút được chọn
        """
        if min_trust is None:
            min_trust = self.config["min_trust_threshold"]
            
        # Lọc các nút hội đủ tiêu chuẩn
        eligible_nodes = {}
        
        for node_id, node_info in self.nodes.items():
            trust_score = self.get_trust_score(node_id)
            
            # Kiểm tra ngưỡng tin cậy
            if trust_score < min_trust:
                continue
                
            # Kiểm tra các thuộc tính bắt buộc
            if required_attributes:
                meets_requirements = True
                for attr, value in required_attributes.items():
                    if attr not in node_info or node_info[attr] != value:
                        meets_requirements = False
                        break
                        
                if not meets_requirements:
                    continue
                    
            # Nút hội đủ tiêu chuẩn
            eligible_nodes[node_id] = trust_score
            
        # Nếu không có đủ nút hội đủ tiêu chuẩn, giảm ngưỡng tin cậy
        if len(eligible_nodes) < num_nodes and min_trust > 0.2:
            return self.select_nodes(num_nodes, min_trust * 0.8, required_attributes)
            
        # Chọn các nút có điểm tin cậy cao nhất
        selected_nodes = sorted(
            eligible_nodes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:num_nodes]
        
        selected_node_ids = [node_id for node_id, _ in selected_nodes]
        
        # Lưu lịch sử lựa chọn
        selection_record = {
            'timestamp': time.time(),
            'selected_nodes': selected_node_ids,
            'num_eligible': len(eligible_nodes),
            'min_trust_used': min_trust
        }
        self.selection_history.append(selection_record)
        
        return selected_node_ids
    
    def select_nodes_probabilistic(self, 
                                 num_nodes: int, 
                                 min_trust: float = None,
                                 temperature: float = 1.0) -> List[str]:
        """
        Lựa chọn các nút với phương pháp xác suất dựa trên điểm tin cậy
        
        Args:
            num_nodes: Số lượng nút cần chọn
            min_trust: Ngưỡng tin cậy tối thiểu
            temperature: Tham số điều chỉnh mức độ ngẫu nhiên (thấp = tất định hơn)
            
        Returns:
            Danh sách ID của các nút được chọn
        """
        if min_trust is None:
            min_trust = self.config["min_trust_threshold"]
            
        # Lọc các nút đủ điều kiện và tính điểm tin cậy
        eligible_nodes = {}
        for node_id in self.nodes.keys():
            trust_score = self.get_trust_score(node_id)
            if trust_score >= min_trust:
                eligible_nodes[node_id] = trust_score
                
        if not eligible_nodes:
            return []
            
        # Chuyển đổi điểm tin cậy thành xác suất (softmax với temperature)
        node_ids = list(eligible_nodes.keys())
        scores = np.array([eligible_nodes[nid] for nid in node_ids])
        probabilities = np.exp(scores / temperature)
        probabilities = probabilities / np.sum(probabilities)
        
        # Lựa chọn ngẫu nhiên dựa trên xác suất (không thay thế)
        num_to_select = min(num_nodes, len(eligible_nodes))
        selected_indices = np.random.choice(
            len(node_ids),
            size=num_to_select,
            replace=False,
            p=probabilities
        )
        
        selected_node_ids = [node_ids[i] for i in selected_indices]
        
        # Lưu lịch sử lựa chọn
        selection_record = {
            'timestamp': time.time(),
            'selected_nodes': selected_node_ids,
            'num_eligible': len(eligible_nodes),
            'selection_method': 'probabilistic',
            'temperature': temperature
        }
        self.selection_history.append(selection_record)
        
        return selected_node_ids
    
    def get_node_ranking(self) -> List[Tuple[str, float]]:
        """
        Lấy xếp hạng của tất cả các nút dựa trên điểm tin cậy
        
        Returns:
            Danh sách các tuple (node_id, trust_score) đã sắp xếp theo điểm giảm dần
        """
        trust_scores = self.get_all_trust_scores()
        ranking = sorted(trust_scores.items(), key=lambda x: x[1], reverse=True)
        return ranking
    
    def get_suspicious_nodes(self, threshold: float = 0.4) -> List[str]:
        """
        Xác định các nút đáng ngờ dựa trên điểm tin cậy thấp
        
        Args:
            threshold: Ngưỡng tin cậy để xác định nút đáng ngờ
            
        Returns:
            Danh sách ID của các nút đáng ngờ
        """
        suspicious = []
        for node_id, node_info in self.nodes.items():
            trust_score = self.get_trust_score(node_id)
            if trust_score < threshold:
                suspicious.append(node_id)
                
        return suspicious
        
    def simulate_trust_evolution(self, 
                               node_metrics: Dict[str, Dict[str, Any]], 
                               steps: int = 10) -> Dict[str, List[float]]:
        """
        Mô phỏng sự tiến hóa điểm tin cậy qua thời gian
        
        Args:
            node_metrics: Từ điển ánh xạ từ node_id đến metrics cho mỗi bước
            steps: Số bước mô phỏng
            
        Returns:
            Từ điển ánh xạ từ node_id đến danh sách điểm tin cậy qua các bước
        """
        # Sao chép trạng thái hiện tại để tránh ảnh hưởng
        original_scores = self.trust_scores.copy()
        original_last_update = self.last_update_time.copy()
        
        # Lưu trữ kết quả mô phỏng
        evolution = defaultdict(list)
        
        for step in range(steps):
            for node_id, metrics in node_metrics.items():
                if node_id in self.nodes:
                    # Thêm nhiễu ngẫu nhiên vào các chỉ số
                    noisy_metrics = {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            noise = np.random.normal(0, 0.05)  # 5% nhiễu
                            noisy_metrics[k] = max(0, v + noise * v)
                        else:
                            noisy_metrics[k] = v
                            
                    # Cập nhật điểm tin cậy với chỉ số có nhiễu
                    score = self.update_node_metrics(node_id, noisy_metrics)
                    evolution[node_id].append(score)
                    
        # Khôi phục trạng thái ban đầu
        self.trust_scores = original_scores
        self.last_update_time = original_last_update
        
        return dict(evolution)
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về hệ thống quản lý tin cậy
        
        Returns:
            Từ điển chứa các thống kê
        """
        trust_scores = self.get_all_trust_scores()
        
        return {
            'num_nodes': len(self.nodes),
            'avg_trust_score': sum(trust_scores.values()) / max(1, len(trust_scores)),
            'min_trust_score': min(trust_scores.values()) if trust_scores else 0,
            'max_trust_score': max(trust_scores.values()) if trust_scores else 0,
            'suspicious_nodes': len(self.get_suspicious_nodes()),
            'selections_made': len(self.selection_history),
            'last_selection': self.selection_history[-1] if self.selection_history else None
        }

    def _collect_network_data(self):
        """Thu thập dữ liệu từ mạng để tính điểm tin cậy"""
        network_data = {}
        
        for shard_id, shard in self.network.shards.items():
            # Thu thập số lượng giao dịch thành công và thất bại
            confirmed = getattr(shard, 'confirmed_count', 0)
            rejected = getattr(shard, 'rejected_count', 0)
            
            # Thu thập thông tin về giao dịch đã xử lý và thất bại
            processed = getattr(shard, 'processed_transactions', {})
            failed = getattr(shard, 'failed_transactions', {})
            
            # Tính tổng số giao dịch
            total_processed = len(processed) if isinstance(processed, dict) else 0
            total_failed = len(failed) if isinstance(failed, dict) else 0
            
            network_data[shard_id] = {
                'confirmed': confirmed,
                'rejected': rejected,
                'total_processed': total_processed,
                'total_failed': total_failed,
                'congestion': getattr(shard, 'congestion_level', 0),
                'latency': getattr(shard, 'avg_latency', 50)
            }
        
        return network_data

    def get_shard_trust_score(self, shard_id):
        """
        Lấy điểm tin cậy của một shard cụ thể
        
        Args:
            shard_id: ID của shard cần lấy điểm tin cậy
            
        Returns:
            Điểm tin cậy của shard, hoặc 0.5 nếu không tìm thấy
        """
        return self.trust_scores.get(shard_id, 0.5)

    def _apply_dqn_actions(self, 
                        node_metrics: Dict[str, Dict[str, Any]], 
                        steps: int = 10) -> Dict[str, List[float]]:
        """
        Mô phỏng sự tiến hóa điểm tin cậy qua thời gian
        
        Args:
            node_metrics: Từ điển ánh xạ từ node_id đến metrics cho mỗi bước
            steps: Số bước mô phỏng
            
        Returns:
            Từ điển ánh xạ từ node_id đến danh sách điểm tin cậy qua các bước
        """
        # Sao chép trạng thái hiện tại để tránh ảnh hưởng
        original_scores = self.trust_scores.copy()
        original_last_update = self.last_update_time.copy()
        
        # Lưu trữ kết quả mô phỏng
        evolution = defaultdict(list)
        
        for step in range(steps):
            for node_id, metrics in node_metrics.items():
                if node_id in self.nodes:
                    # Thêm nhiễu ngẫu nhiên vào các chỉ số
                    noisy_metrics = {}
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            noise = np.random.normal(0, 0.05)  # 5% nhiễu
                            noisy_metrics[k] = max(0, v + noise * v)
                        else:
                            noisy_metrics[k] = v
                            
                    # Cập nhật điểm tin cậy với chỉ số có nhiễu
                    score = self.update_node_metrics(node_id, noisy_metrics)
                    evolution[node_id].append(score)
                    
        # Khôi phục trạng thái ban đầu
        self.trust_scores = original_scores
        self.last_update_time = original_last_update
        
        return dict(evolution)

    def update_trust_scores(self):
        """
        Cập nhật điểm tin cậy cho tất cả các nút dựa trên dữ liệu mạng hiện tại
        """
        # Thu thập dữ liệu mạng
        network_data = self._collect_network_data()
        
        # Đảm bảo khởi tạo cấu trúc dữ liệu cần thiết
        if not hasattr(self, 'node_reliability'):
            self.node_reliability = {}
        
        if not hasattr(self, 'shard_reliability'):
            self.shard_reliability = {}
            
        if not hasattr(self, 'cross_shard_trust'):
            self.cross_shard_trust = {}
            
        if not hasattr(self, 'trust_violations'):
            self.trust_violations = 0
        
        # Cập nhật điểm tin cậy cho từng nút
        for node_id, node_info in self.nodes.items():
            # Tạo metrics từ dữ liệu mạng
            metrics = {
                "uptime": node_info.get("uptime", 1.0),
                "performance": node_info.get("compute_power", 0.5) * (1.0 + random.uniform(-0.1, 0.1)),  # Thêm một chút nhiễu
                "security_incidents": network_data.get(node_id, {}).get("security_incidents", 0),
                "energy_efficiency": node_info.get("energy_efficiency", 0.7) * (1.0 + random.uniform(-0.05, 0.05)),
                "successful_validations": network_data.get(node_id, {}).get("successful_validations", 10),
                "failed_validations": network_data.get(node_id, {}).get("failed_validations", 0),
                "latency": network_data.get(node_id, {}).get("latency", 50) / 1000  # chuyển từ ms sang giây
            }
            
            # Cập nhật độ tin cậy của node dựa trên hoạt động của nó
            if metrics["successful_validations"] + metrics["failed_validations"] > 0:
                reliability = metrics["successful_validations"] / (metrics["successful_validations"] + metrics["failed_validations"])
                self.node_reliability[node_id] = reliability
                
                # Phát hiện vi phạm độ tin cậy
                if reliability < 0.5 and metrics["failed_validations"] > 5:
                    self.trust_violations += 1
            
            # Cập nhật điểm tin cậy
            self.update_node_metrics(node_id, metrics)
        
        # Cập nhật điểm tin cậy cho các shard
        if self.network and hasattr(self.network, 'shards'):
            for shard_id in self.network.shards:
                # Tính điểm tin cậy trung bình của các nút trong shard
                shard_nodes = [n for n, info in self.nodes.items() 
                              if info.get('shard_id') == shard_id]
                
                if shard_nodes:
                    avg_score = sum(self.trust_scores.get(n, 0.5) for n in shard_nodes) / len(shard_nodes)
                    self.trust_scores[shard_id] = avg_score
                    
                    # Tính độ tin cậy của shard dựa trên hiệu suất và số giao dịch thành công
                    shard = self.network.shards[shard_id]
                    success_rate = getattr(shard, 'successful_cs_tx_count', 0) / max(1, getattr(shard, 'cross_shard_tx_count', 1))
                    reliability = (avg_score + success_rate) / 2  # Kết hợp độ tin cậy của nodes và tỷ lệ thành công
                    self.shard_reliability[shard_id] = reliability
        
            # Tính điểm tin cậy giữa các cặp shard
            self._update_cross_shard_trust()

    def _update_cross_shard_trust(self):
        """
        Cập nhật ma trận độ tin cậy giữa các cặp shard
        """
        if not self.network or not hasattr(self.network, 'shards'):
            return
            
        # Khởi tạo ma trận độ tin cậy
        for i in self.network.shards:
            if i not in self.cross_shard_trust:
                self.cross_shard_trust[i] = {}
                
            for j in self.network.shards:
                if i == j:
                    # Độ tin cậy trong cùng một shard luôn cao
                    self.cross_shard_trust[i][j] = 1.0
                elif j not in self.cross_shard_trust[i]:
                    # Khởi tạo với giá trị mặc định
                    self.cross_shard_trust[i][j] = 0.7
        
        # Cập nhật ma trận độ tin cậy dựa trên dữ liệu lịch sử giao dịch
        for i in self.network.shards:
            for j in self.network.shards:
                if i == j:
                    continue
                    
                # Lấy dữ liệu từ hai shard
                shard_i = self.network.shards[i]
                shard_j = self.network.shards[j]
                
                # Đánh giá độ tin cậy dựa trên số lượng giao dịch thành công và thất bại
                successful_tx = 0
                failed_tx = 0
                
                # Lấy giao dịch từ shard i đến shard j
                if hasattr(shard_i, 'processed_transactions') and hasattr(shard_i, 'failed_transactions'):
                    for tx_id, tx in getattr(shard_i, 'processed_transactions', {}).items():
                        if hasattr(tx, 'target_shard') and tx.target_shard == j:
                            successful_tx += 1
                            
                    for tx_id, tx in getattr(shard_i, 'failed_transactions', {}).items():
                        if hasattr(tx, 'target_shard') and tx.target_shard == j:
                            failed_tx += 1
                
                # Tính độ tin cậy mới
                if successful_tx + failed_tx > 0:
                    trust_ij = successful_tx / (successful_tx + failed_tx)
                    # Cập nhật với decay rate để giữ lại một phần thông tin cũ
                    decay_rate = 0.3
                    self.cross_shard_trust[i][j] = (1 - decay_rate) * self.cross_shard_trust[i][j] + decay_rate * trust_ij
                    
                    # Đảm bảo tính đối xứng một phần
                    if j in self.cross_shard_trust and i in self.cross_shard_trust[j]:
                        trust_ji = self.cross_shard_trust[j][i]
                        # Cập nhật trust_ji dựa một phần vào trust_ij
                        self.cross_shard_trust[j][i] = 0.8 * trust_ji + 0.2 * trust_ij

    def get_trust_between_shards(self, source_shard_id, target_shard_id):
        """
        Lấy mức độ tin cậy giữa hai shard
        
        Args:
            source_shard_id: ID của shard nguồn
            target_shard_id: ID của shard đích
            
        Returns:
            float: Mức độ tin cậy trong khoảng [0, 1]
        """
        if not hasattr(self, 'cross_shard_trust'):
            self._update_cross_shard_trust()
            
        if source_shard_id in self.cross_shard_trust and target_shard_id in self.cross_shard_trust[source_shard_id]:
            return self.cross_shard_trust[source_shard_id][target_shard_id]
        
        # Giá trị mặc định cho trường hợp chưa có dữ liệu
        return 0.7
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Trả về thống kê về quản lý tin cậy
        
        Returns:
            Dict: Thống kê về quản lý tin cậy
        """
        stats = super().get_statistics() if hasattr(super(), 'get_statistics') else {}
        
        # Thống kê tổng quát
        stats.update({
            "total_nodes": len(self.nodes),
            "avg_trust_score": sum(self.trust_scores.values()) / max(1, len(self.trust_scores)),
            "min_trust_score": min(self.trust_scores.values()) if self.trust_scores else 0,
            "max_trust_score": max(self.trust_scores.values()) if self.trust_scores else 0,
            "trust_violations_detected": getattr(self, 'trust_violations', 0),
            "shard_reliability": getattr(self, 'shard_reliability', {}),
            "transaction_security_level": {
                "high": len([s for s in getattr(self, 'shard_reliability', {}).values() if s > 0.8]),
                "medium": len([s for s in getattr(self, 'shard_reliability', {}).values() if 0.5 <= s <= 0.8]),
                "low": len([s for s in getattr(self, 'shard_reliability', {}).values() if s < 0.5]),
            }
        })
        
        return stats