"""
Module triển khai học liên kết phân tán cho việc tối ưu hóa blockchain
"""

import copy
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
import random
from collections import defaultdict

from dqn_blockchain_sim.configs.simulation_config import FL_CONFIG


class DifferentialPrivacy:
    """
    Lớp cung cấp các phương pháp bảo vệ tính riêng tư vi phân
    """
    
    @staticmethod
    def add_noise(gradients: Dict[str, torch.Tensor], 
                 noise_scale: float, 
                 clip_norm: float) -> Dict[str, torch.Tensor]:
        """
        Thêm nhiễu Gaussian vào gradient để bảo vệ tính riêng tư
        
        Args:
            gradients: Từ điển chứa các gradient
            noise_scale: Độ lớn của nhiễu (sigma)
            clip_norm: Ngưỡng cắt cho các gradient
            
        Returns:
            Gradient đã được thêm nhiễu
        """
        # Sao chép gradient để không ảnh hưởng đến bản gốc
        noisy_gradients = copy.deepcopy(gradients)
        
        # Cắt gradient
        total_norm = 0
        for name, grad in noisy_gradients.items():
            param_norm = grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for name, grad in noisy_gradients.items():
                grad.data.mul_(clip_coef)
                
        # Thêm nhiễu Gaussian
        for name, grad in noisy_gradients.items():
            noise = torch.randn_like(grad) * noise_scale
            grad.data.add_(noise)
            
        return noisy_gradients


class SecureAggregation:
    """
    Lớp cung cấp các phương pháp tổng hợp an toàn cho học liên kết
    """
    
    @staticmethod
    def aggregate_parameters(parameters_list: List[Dict[str, torch.Tensor]], 
                           weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp các tham số mô hình từ nhiều máy khách
        
        Args:
            parameters_list: Danh sách các từ điển tham số từ các máy khách
            weights: Trọng số cho mỗi máy khách (mặc định là đều nhau)
            
        Returns:
            Từ điển chứa các tham số đã tổng hợp
        """
        if not parameters_list:
            return {}
            
        # Nếu không có trọng số, sử dụng trọng số đều nhau
        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)
            
        # Chuẩn hóa trọng số
        weight_sum = sum(weights)
        if weight_sum > 0:
            weights = [w / weight_sum for w in weights]
            
        # Khởi tạo từ điển kết quả với tham số từ khách hàng đầu tiên
        result = {}
        for name, param in parameters_list[0].items():
            result[name] = param.clone() * weights[0]
            
        # Tổng hợp tham số từ các khách hàng còn lại
        for client_idx in range(1, len(parameters_list)):
            client_params = parameters_list[client_idx]
            client_weight = weights[client_idx]
            
            for name, param in client_params.items():
                if name in result:
                    result[name].data.add_(param.data * client_weight)
                    
        return result
    
    @staticmethod
    def secure_aggregation_with_dropout(parameters_list: List[Dict[str, torch.Tensor]], 
                                      weights: List[float] = None,
                                      dropout_rate: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Tổng hợp an toàn với khả năng chống chịu với việc dropout của máy khách
        
        Args:
            parameters_list: Danh sách các từ điển tham số từ các máy khách
            weights: Trọng số cho mỗi máy khách
            dropout_rate: Tỷ lệ máy khách có thể bị ngắt kết nối
            
        Returns:
            Từ điển chứa các tham số đã tổng hợp
        """
        if not parameters_list:
            return {}
            
        # Mô phỏng dropout
        if dropout_rate > 0:
            client_indices = list(range(len(parameters_list)))
            keep_indices = random.sample(
                client_indices, 
                int(len(client_indices) * (1 - dropout_rate))
            )
            parameters_list = [parameters_list[i] for i in keep_indices]
            if weights is not None:
                weights = [weights[i] for i in keep_indices]
                
        # Tiến hành tổng hợp với danh sách đã lọc
        return SecureAggregation.aggregate_parameters(parameters_list, weights)


class FederatedLearner:
    """
    Lớp chính để triển khai học liên kết phân tán
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo bộ học liên kết
        
        Args:
            config: Cấu hình FL, sử dụng mặc định nếu không cung cấp
        """
        self.config = config if config is not None else FL_CONFIG
        self.round = 0
        self.clients = {}  # Ánh xạ từ client_id đến thông tin client
        self.global_model = None  # Mô hình toàn cục
        self.selected_clients = []  # Danh sách khách hàng được chọn cho vòng hiện tại
        self.aggregation_history = []  # Lịch sử tổng hợp
        
    def add_client(self, client_id: str, client_info: Dict[str, Any]) -> None:
        """
        Thêm một khách hàng vào hệ thống học liên kết
        
        Args:
            client_id: ID của khách hàng
            client_info: Thông tin khách hàng (metadata)
        """
        self.clients[client_id] = {
            'info': client_info,
            'last_update': 0,  # Vòng cập nhật cuối cùng
            'reliability': 1.0,  # Độ tin cậy ban đầu
            'local_updates': 0,  # Số lần cập nhật cục bộ
            'trust_score': 0.9  # Điểm tin cậy ban đầu
        }
        
    def remove_client(self, client_id: str) -> bool:
        """
        Xóa một khách hàng khỏi hệ thống học liên kết
        
        Args:
            client_id: ID của khách hàng cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if client_id in self.clients:
            del self.clients[client_id]
            if client_id in self.selected_clients:
                self.selected_clients.remove(client_id)
            return True
        return False
        
    def select_clients(self) -> List[str]:
        """
        Chọn một tập hợp khách hàng cho vòng học liên kết hiện tại
        
        Returns:
            Danh sách ID của các khách hàng được chọn
        """
        # Đảm bảo có đủ khách hàng
        if len(self.clients) < self.config['min_clients']:
            self.selected_clients = []
            return []
            
        # Số lượng khách hàng cần chọn
        num_to_select = max(
            self.config['min_clients'],
            int(len(self.clients) * self.config['client_fraction'])
        )
        
        # Tính toán trọng số lựa chọn dựa trên độ tin cậy
        selection_weights = {}
        for client_id, client_data in self.clients.items():
            # Trọng số dựa trên độ tin cậy và thời gian từ lần cập nhật cuối
            time_factor = (self.round - client_data['last_update']) / max(1, self.round)
            selection_weights[client_id] = client_data['trust_score'] * (1 + time_factor)
            
        # Chọn khách hàng dựa trên trọng số
        client_ids = list(self.clients.keys())
        weights = [selection_weights[cid] for cid in client_ids]
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(client_ids)] * len(client_ids)
            
        # Lựa chọn có trọng số không thay thế
        self.selected_clients = random.choices(
            client_ids, 
            weights=weights, 
            k=min(num_to_select, len(client_ids))
        )
        
        return self.selected_clients
    
    def initialize_global_model(self, model_parameters: Dict[str, torch.Tensor]) -> None:
        """
        Khởi tạo mô hình toàn cục với tham số ban đầu
        
        Args:
            model_parameters: Từ điển chứa các tham số mô hình ban đầu
        """
        self.global_model = copy.deepcopy(model_parameters)
        
    def get_model_for_client(self, client_id: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Lấy mô hình hiện tại cho một khách hàng cụ thể
        
        Args:
            client_id: ID của khách hàng
            
        Returns:
            Từ điển chứa các tham số mô hình hoặc None nếu không hợp lệ
        """
        if client_id not in self.clients or self.global_model is None:
            return None
            
        # Trả về bản sao của mô hình toàn cục
        return copy.deepcopy(self.global_model)
    
    def receive_client_update(self, client_id: str, 
                             model_update: Dict[str, torch.Tensor],
                             metrics: Dict[str, Any] = None) -> bool:
        """
        Nhận cập nhật mô hình từ một khách hàng
        
        Args:
            client_id: ID của khách hàng
            model_update: Từ điển chứa các tham số mô hình đã cập nhật
            metrics: Các chỉ số hiệu suất từ khách hàng (tùy chọn)
            
        Returns:
            True nếu cập nhật hợp lệ, False nếu không
        """
        if client_id not in self.clients or client_id not in self.selected_clients:
            return False
            
        # Cập nhật thông tin khách hàng
        self.clients[client_id]['last_update'] = self.round
        self.clients[client_id]['local_updates'] += 1
        
        # Lưu trữ cập nhật mô hình với metadata
        if not hasattr(self, 'current_round_updates'):
            self.current_round_updates = []
            
        # Áp dụng tính riêng tư vi phân nếu được cấu hình
        if self.config.get('dp_epsilon', 0) > 0:
            # Tính toán độ lớn nhiễu dựa trên epsilon và delta
            noise_scale = np.sqrt(2 * np.log(1.25 / self.config['dp_delta'])) / self.config['dp_epsilon']
            
            # Thêm nhiễu với tính riêng tư vi phân
            model_update = DifferentialPrivacy.add_noise(
                model_update,
                noise_scale,
                self.config['dp_clip_norm']
            )
            
        # Tính toán trọng số cho khách hàng này
        client_weight = self.clients[client_id]['trust_score']
        
        # Lưu trữ cập nhật và trọng số
        self.current_round_updates.append({
            'client_id': client_id,
            'parameters': model_update,
            'weight': client_weight,
            'metrics': metrics
        })
        
        # Cập nhật điểm tin cậy nếu có metrics
        if metrics is not None:
            self._update_client_trust_score(client_id, metrics)
            
        return True
    
    def _update_client_trust_score(self, client_id: str, metrics: Dict[str, Any]) -> None:
        """
        Cập nhật điểm tin cậy của khách hàng dựa trên các chỉ số hiệu suất
        
        Args:
            client_id: ID của khách hàng
            metrics: Các chỉ số hiệu suất từ khách hàng
        """
        if client_id not in self.clients:
            return
            
        # Thuật toán đánh giá tin cậy đơn giản dựa trên hiệu suất mô hình
        current_score = self.clients[client_id]['trust_score']
        
        # Các yếu tố ảnh hưởng đến điểm tin cậy
        loss_factor = 1.0
        anomaly_factor = 1.0
        
        # Điều chỉnh dựa trên hàm mất mát (loss thấp -> tin cậy cao)
        if 'loss' in metrics:
            norm_loss = min(1.0, metrics['loss'] / 10.0)  # Chuẩn hóa giá trị loss
            loss_factor = 1.0 - 0.3 * norm_loss
            
        # Phát hiện các giá trị bất thường trong mô hình
        if 'gradient_norm' in metrics:
            # Gradient quá lớn có thể là dấu hiệu của tấn công
            norm_grad = min(1.0, metrics['gradient_norm'] / 100.0)
            anomaly_factor = 1.0 - 0.5 * (norm_grad ** 2)
            
        # Điều chỉnh điểm tin cậy
        new_score = current_score * 0.8 + 0.2 * (loss_factor * anomaly_factor)
        
        # Đảm bảo điểm tin cậy nằm trong khoảng [0, 1]
        new_score = max(0.1, min(1.0, new_score))
        
        # Cập nhật điểm tin cậy
        self.clients[client_id]['trust_score'] = new_score
    
    def aggregate_updates(self) -> bool:
        """
        Tổng hợp các cập nhật từ các khách hàng đã chọn
        
        Returns:
            True nếu tổng hợp thành công, False nếu không
        """
        if not hasattr(self, 'current_round_updates') or not self.current_round_updates:
            return False
            
        # Lấy danh sách tham số và trọng số
        parameters_list = [update['parameters'] for update in self.current_round_updates]
        weights = [update['weight'] for update in self.current_round_updates]
        
        # Tổng hợp tham số dựa trên phương pháp được cấu hình
        aggregation_method = self.config.get('aggregation_method', 'fedavg_secure')
        
        if aggregation_method == 'fedavg':
            # FedAvg cơ bản
            aggregated_params = SecureAggregation.aggregate_parameters(parameters_list, weights)
            
        elif aggregation_method == 'fedavg_secure':
            # FedAvg với khả năng chống chịu dropout
            dropout_rate = 0.1  # Tỷ lệ giả định client có thể dropout
            aggregated_params = SecureAggregation.secure_aggregation_with_dropout(
                parameters_list, weights, dropout_rate)
            
        else:
            # Mặc định sử dụng FedAvg cơ bản
            aggregated_params = SecureAggregation.aggregate_parameters(parameters_list, weights)
            
        # Cập nhật mô hình toàn cục
        if self.global_model is None:
            self.global_model = aggregated_params
        else:
            # Cập nhật mô hình toàn cục với tham số đã tổng hợp
            for name, param in aggregated_params.items():
                if name in self.global_model:
                    self.global_model[name].copy_(param)
                    
        # Lưu lịch sử tổng hợp
        aggregation_info = {
            'round': self.round,
            'num_clients': len(self.current_round_updates),
            'client_ids': [update['client_id'] for update in self.current_round_updates],
            'timestamp': np.datetime64('now')
        }
        self.aggregation_history.append(aggregation_info)
        
        # Xóa cập nhật hiện tại
        del self.current_round_updates
        
        return True
    
    def step(self) -> Tuple[int, List[str]]:
        """
        Thực hiện một vòng học liên kết, bao gồm lựa chọn khách hàng,
        nhận cập nhật, và tổng hợp mô hình
        
        Returns:
            Tuple (round, selected_clients)
        """
        # Tăng số vòng
        self.round += 1
        
        # Chọn khách hàng cho vòng này
        selected_clients = self.select_clients()
        
        # Tổng hợp cập nhật từ vòng trước nếu đến thời điểm tổng hợp và có đủ cập nhật
        if (self.round % self.config['aggregation_frequency'] == 0 and
            hasattr(self, 'current_round_updates') and 
            len(self.current_round_updates) >= self.config['min_clients']):
            self.aggregate_updates()
            
        return self.round, selected_clients
    
    def get_client_trust_scores(self) -> Dict[str, float]:
        """
        Lấy điểm tin cậy của tất cả các khách hàng
        
        Returns:
            Từ điển ánh xạ từ client_id đến điểm tin cậy
        """
        return {cid: data['trust_score'] for cid, data in self.clients.items()}
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """
        Lấy mô hình toàn cục hiện tại
        
        Returns:
            Từ điển chứa các tham số mô hình toàn cục hoặc None nếu chưa khởi tạo
        """
        return copy.deepcopy(self.global_model) if self.global_model is not None else None
    
    def detect_anomalies(self) -> List[str]:
        """
        Phát hiện các khách hàng bất thường hoặc độc hại
        
        Returns:
            Danh sách ID của các khách hàng bị nghi ngờ
        """
        suspicious_clients = []
        trust_threshold = 0.5  # Ngưỡng tin cậy tối thiểu
        
        for client_id, client_data in self.clients.items():
            # Kiểm tra điểm tin cậy
            if client_data['trust_score'] < trust_threshold:
                suspicious_clients.append(client_id)
                
            # Kiểm tra thời gian không hoạt động
            if self.round - client_data['last_update'] > 10:
                # Khách hàng không tham gia trong 10 vòng
                if client_id not in suspicious_clients:
                    suspicious_clients.append(client_id)
                    
        return suspicious_clients
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê về quá trình học liên kết
        
        Returns:
            Từ điển chứa các thống kê
        """
        return {
            'round': self.round,
            'num_clients': len(self.clients),
            'selected_clients': len(self.selected_clients),
            'aggregation_history': len(self.aggregation_history),
            'avg_trust_score': sum(self.get_client_trust_scores().values()) / max(1, len(self.clients)),
            'suspicious_clients': len(self.detect_anomalies()),
            'last_aggregation': self.aggregation_history[-1] if self.aggregation_history else None
        } 