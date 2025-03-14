"""
Module học liên kết (Federated Learning) cho các DQN agent trong blockchain
Module này triển khai cơ chế chia sẻ tri thức giữa các DQN agent trên các shard
khác nhau để cải thiện tốc độ hội tụ và hiệu suất tổng thể.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Set
import copy
import time
from collections import defaultdict, deque
import threading
import random

class FederatedLearning:
    """
    Lớp học liên kết (Federated Learning) để tổng hợp các mô hình từ DQN agents.
    """
    
    def __init__(self, 
                global_rounds: int = 5, 
                local_epochs: int = 2,
                min_clients: int = 2,
                client_fraction: float = 0.8,
                aggregation_method: str = "fedavg",
                secure_aggregation: bool = True,
                privacy_budget: float = 1.0,
                log_dir: str = "logs"):
        """
        Khởi tạo học liên kết
        
        Args:
            global_rounds: Số vòng huấn luyện toàn cục
            local_epochs: Số epoch huấn luyện cục bộ trước mỗi tổng hợp
            min_clients: Số lượng client tối thiểu để thực hiện tổng hợp
            client_fraction: Tỷ lệ client tham gia mỗi vòng huấn luyện
            aggregation_method: Phương pháp tổng hợp ("fedavg", "fedprox", "fedadam")
            secure_aggregation: Có sử dụng tổng hợp an toàn không
            privacy_budget: Ngân sách bảo mật cho differential privacy
            log_dir: Thư mục lưu log
        """
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.min_clients = min_clients
        self.client_fraction = client_fraction
        self.aggregation_method = aggregation_method
        self.secure_aggregation = secure_aggregation
        self.privacy_budget = privacy_budget
        self.log_dir = log_dir
        
        # Lưu trữ client models và metrics
        self.clients = {}
        self.global_model = None
        self.global_model_version = 0
        self.aggregation_metrics = defaultdict(list)
        
        # Khóa cho đồng bộ hóa
        self.aggregation_lock = threading.Lock()
        
        # Điểm chuẩn hiệu suất
        self.performance_benchmark = {
            "aggregation_time": [],
            "communication_overhead": [],
            "model_improvements": []
        }
        
    def register_client(self, client_id: str, model: nn.Module, data_size: int = 0):
        """
        Đăng ký một client mới cho federated learning
        
        Args:
            client_id: ID của client (shard)
            model: Mô hình DQN của client
            data_size: Kích thước dữ liệu cục bộ (dùng để tính trọng số)
        """
        if client_id in self.clients:
            print(f"Client {client_id} đã được đăng ký. Cập nhật mô hình mới.")
        
        self.clients[client_id] = {
            "model": copy.deepcopy(model),
            "data_size": data_size,
            "last_update": time.time(),
            "updates_count": 0,
            "performance_history": []
        }
        
        # Lưu đối tượng client gốc để sử dụng sau này
        if hasattr(model, "get_client_instance"):
            self.clients[client_id]["client_instance"] = model.get_client_instance()
        
        # Nếu chưa có mô hình toàn cục, khởi tạo nó với mô hình client đầu tiên
        if self.global_model is None:
            self.global_model = copy.deepcopy(model)
            self.global_model_version = 1
            print(f"Đã khởi tạo mô hình toàn cục từ client {client_id}")
            
    def update_client_model(self, client_id: str, model: nn.Module, metrics: Dict[str, Any] = None):
        """
        Cập nhật mô hình cục bộ của client sau khi huấn luyện
        
        Args:
            client_id: ID của client
            model: Mô hình đã cập nhật của client
            metrics: Các chỉ số hiệu suất từ việc huấn luyện
        """
        if client_id not in self.clients:
            raise ValueError(f"Client {client_id} chưa được đăng ký")
        
        # Cập nhật mô hình
        self.clients[client_id]["model"] = copy.deepcopy(model)
        self.clients[client_id]["last_update"] = time.time()
        self.clients[client_id]["updates_count"] += 1
        
        # Cập nhật chỉ số nếu có
        if metrics:
            self.clients[client_id]["performance_history"].append(metrics)
            
        # Ghi log
        print(f"Đã cập nhật mô hình cho client {client_id}. "
              f"Tổng số cập nhật: {self.clients[client_id]['updates_count']}")
    
    def select_clients(self) -> List[str]:
        """
        Chọn tập hợp con các client để tham gia vào quá trình huấn luyện
        
        Returns:
            Danh sách ID client được chọn
        """
        if len(self.clients) < self.min_clients:
            return []
            
        num_clients = max(int(len(self.clients) * self.client_fraction), self.min_clients)
        selected_clients = random.sample(list(self.clients.keys()), num_clients)
        
        return selected_clients
        
    def aggregate_models(self, client_ids: List[str] = None) -> nn.Module:
        """
        Tổng hợp các mô hình từ clients được chọn
        
        Args:
            client_ids: Danh sách ID client để tổng hợp, nếu None thì chọn ngẫu nhiên
            
        Returns:
            Mô hình tổng hợp toàn cục
        """
        with self.aggregation_lock:
            start_time = time.time()
            
            # Chọn clients nếu không được chỉ định
            if client_ids is None:
                client_ids = self.select_clients()
                
            if not client_ids:
                print("Không đủ clients để thực hiện tổng hợp")
                return self.global_model
                
            print(f"Tổng hợp mô hình từ {len(client_ids)} clients: {client_ids}")
            
            # Tính tổng trọng số
            total_data_size = sum(self.clients[client_id]["data_size"] for client_id in client_ids)
            if total_data_size == 0:
                # Nếu không có thông tin kích thước dữ liệu, sử dụng trung bình cộng
                weights = {client_id: 1.0 / len(client_ids) for client_id in client_ids}
            else:
                # Trọng số tỷ lệ với kích thước dữ liệu
                weights = {client_id: self.clients[client_id]["data_size"] / total_data_size 
                          for client_id in client_ids}
            
            # Tạo bản sao của mô hình toàn cục để cập nhật
            new_global_model = copy.deepcopy(self.global_model)
            
            # Áp dụng phương pháp tổng hợp thích hợp
            if self.aggregation_method == "fedavg":
                self._fedavg_aggregation(new_global_model, client_ids, weights)
            elif self.aggregation_method == "fedprox":
                self._fedprox_aggregation(new_global_model, client_ids, weights)
            elif self.aggregation_method == "fedadam":
                self._fedadam_aggregation(new_global_model, client_ids, weights)
            else:
                self._fedavg_aggregation(new_global_model, client_ids, weights)
                
            # Cập nhật mô hình toàn cục
            self.global_model = new_global_model
            self.global_model_version += 1
            
            # Ghi nhận thời gian tổng hợp
            aggregation_time = time.time() - start_time
            self.performance_benchmark["aggregation_time"].append(aggregation_time)
            
            print(f"Tổng hợp mô hình hoàn tất. Phiên bản mô hình toàn cục: {self.global_model_version}")
            print(f"Thời gian tổng hợp: {aggregation_time:.4f} giây")
            
            return self.global_model
            
    def _fedavg_aggregation(self, global_model: nn.Module, client_ids: List[str], weights: Dict[str, float]):
        """
        Thực hiện tổng hợp FedAvg (Federated Averaging)
        
        Args:
            global_model: Mô hình toàn cục đang cập nhật
            client_ids: Danh sách ID client tham gia
            weights: Trọng số cho mỗi client
        """
        # Lấy trạng thái từ điển của mô hình toàn cục
        global_state_dict = global_model.state_dict()
        
        # Khởi tạo các tham số trung bình có trọng số
        for key in global_state_dict.keys():
            # Đặt lại tham số về 0
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Tính tổng có trọng số
            for client_id in client_ids:
                client_model = self.clients[client_id]["model"]
                client_weight = weights[client_id]
                
                # Thực hiện tổng có trọng số
                global_state_dict[key] += client_weight * client_model.state_dict()[key]
        
        # Tải trạng thái từ điển đã tổng hợp vào mô hình toàn cục
        global_model.load_state_dict(global_state_dict)
        
    def _fedprox_aggregation(self, global_model: nn.Module, client_ids: List[str], weights: Dict[str, float], mu: float = 0.01):
        """
        Thực hiện tổng hợp FedProx (Federated Proximal)
        
        Args:
            global_model: Mô hình toàn cục đang cập nhật
            client_ids: Danh sách ID client tham gia
            weights: Trọng số cho mỗi client
            mu: Tham số cân bằng gần đúng
        """
        # Lưu trạng thái từ điển ban đầu của mô hình toàn cục
        initial_global_state = copy.deepcopy(global_model.state_dict())
        global_state_dict = global_model.state_dict()
        
        # Khởi tạo tham số trung bình có trọng số
        for key in global_state_dict.keys():
            # Đặt lại về 0
            global_state_dict[key] = torch.zeros_like(global_state_dict[key])
            
            # Tính tổng có trọng số với thuật ngữ cập nhật gần đúng
            for client_id in client_ids:
                client_model = self.clients[client_id]["model"]
                client_weight = weights[client_id]
                
                # FedProx update: w_i + µ(w_i - w_g)
                client_param = client_model.state_dict()[key]
                proximal_term = client_param + mu * (client_param - initial_global_state[key])
                
                global_state_dict[key] += client_weight * proximal_term
        
        # Tải trạng thái từ điển đã tổng hợp
        global_model.load_state_dict(global_state_dict)
        
    def _fedadam_aggregation(self, global_model: nn.Module, client_ids: List[str], weights: Dict[str, float], 
                           beta1: float = 0.9, beta2: float = 0.99, eta: float = 0.01):
        """
        Thực hiện tổng hợp FedAdam (Federated Adam Optimizer)
        
        Args:
            global_model: Mô hình toàn cục đang cập nhật
            client_ids: Danh sách ID client tham gia
            weights: Trọng số cho mỗi client
            beta1: Tham số động lượng
            beta2: Tham số vận tốc bình phương
            eta: Tốc độ học
        """
        # Khởi tạo biến động lượng và vận tốc bình phương nếu chưa có
        if not hasattr(self, "m"):
            self.m = {}
            self.v = {}
            self.t = 0
            
            for key in global_model.state_dict().keys():
                self.m[key] = torch.zeros_like(global_model.state_dict()[key])
                self.v[key] = torch.zeros_like(global_model.state_dict()[key])
                
        self.t += 1
        global_state_dict = global_model.state_dict()
        
        # Tính delta cho mỗi tham số (sự khác biệt trung bình có trọng số)
        delta = {}
        for key in global_state_dict.keys():
            delta[key] = torch.zeros_like(global_state_dict[key])
            
            # Tính trung bình có trọng số của các cập nhật từ clients
            weighted_client_params = torch.zeros_like(global_state_dict[key])
            for client_id in client_ids:
                client_model = self.clients[client_id]["model"]
                client_weight = weights[client_id]
                weighted_client_params += client_weight * client_model.state_dict()[key]
                
            # Tính delta so với mô hình toàn cục hiện tại
            delta[key] = weighted_client_params - global_state_dict[key]
            
            # Cập nhật động lượng và vận tốc bình phương
            self.m[key] = beta1 * self.m[key] + (1 - beta1) * delta[key]
            self.v[key] = beta2 * self.v[key] + (1 - beta2) * (delta[key] ** 2)
            
            # Hiệu chỉnh thiên lệch
            m_hat = self.m[key] / (1 - beta1 ** self.t)
            v_hat = self.v[key] / (1 - beta2 ** self.t)
            
            # Cập nhật tham số toàn cục
            update = eta * m_hat / (torch.sqrt(v_hat) + 1e-8)
            global_state_dict[key] += update
        
        # Tải trạng thái từ điển đã cập nhật
        global_model.load_state_dict(global_state_dict)
            
    def distribute_global_model(self, client_ids: List[str] = None):
        """
        Phân phối mô hình toàn cục cho các client
        
        Args:
            client_ids: Danh sách ID client để phân phối, nếu None thì phân phối cho tất cả
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())
            
        for client_id in client_ids:
            if client_id in self.clients:
                # Tạo một bản sao của mô hình toàn cục cho client
                self.clients[client_id]["model"] = copy.deepcopy(self.global_model)
                self.clients[client_id]["last_update"] = time.time()
                
                # Cập nhật instance
                if "client_instance" in self.clients[client_id]:
                    client_instance = self.clients[client_id]["client_instance"]
                    if hasattr(client_instance, "update_model_from_global"):
                        client_instance.update_model_from_global(self.global_model)
                        
                print(f"Đã phân phối mô hình toàn cục v{self.global_model_version} cho client {client_id}")
                
    def apply_differential_privacy(self, model: nn.Module, noise_scale: float = 0.01) -> nn.Module:
        """
        Áp dụng differential privacy vào mô hình
        
        Args:
            model: Mô hình đầu vào
            noise_scale: Tỷ lệ nhiễu
            
        Returns:
            Mô hình với nhiễu differential privacy
        """
        if self.privacy_budget <= 0:
            return model
            
        # Tạo bản sao để không ảnh hưởng đến model gốc
        model_copy = copy.deepcopy(model)
        state_dict = model_copy.state_dict()
        
        # Thêm nhiễu Gaussian vào mỗi tham số
        for key in state_dict:
            if state_dict[key].dtype == torch.float32 or state_dict[key].dtype == torch.float64:
                noise = torch.randn_like(state_dict[key]) * noise_scale * self.privacy_budget
                state_dict[key] += noise
                
        # Tải trạng thái từ điển đã thêm nhiễu
        model_copy.load_state_dict(state_dict)
        return model_copy
        
    def secure_aggregate(self, models: Dict[str, nn.Module]) -> nn.Module:
        """
        Thực hiện tổng hợp an toàn để bảo vệ quyền riêng tư client
        
        Args:
            models: Từ điển mô hình của client (ID : model)
            
        Returns:
            Mô hình tổng hợp an toàn
        """
        if not self.secure_aggregation or not models:
            return None
            
        # Lấy mô hình đầu tiên làm template
        template_model = next(iter(models.values()))
        secure_model = copy.deepcopy(template_model)
        
        # Lưu trữ tensors để tổng hợp
        tensor_dict = defaultdict(list)
        
        # Thu thập tensors từ mỗi mô hình
        for client_id, model in models.items():
            for key, param in model.state_dict().items():
                tensor_dict[key].append(param)
                
        # Tổng hợp an toàn bằng cách thêm nhiễu ngẫu nhiên
        secure_state_dict = secure_model.state_dict()
        for key in secure_state_dict:
            if key in tensor_dict:
                # Kiểm tra và chuyển đổi kiểu dữ liệu nếu cần
                tensors_to_stack = []
                for tensor in tensor_dict[key]:
                    # Chuyển đổi Long tensor thành Float tensor
                    if tensor.dtype == torch.int64 or tensor.dtype == torch.long:
                        tensors_to_stack.append(tensor.float())
                    else:
                        tensors_to_stack.append(tensor)
                
                # Tính giá trị trung bình
                mean_tensor = torch.stack(tensors_to_stack).mean(dim=0)
                
                # Chuyển về kiểu dữ liệu ban đầu nếu cần
                if secure_state_dict[key].dtype == torch.int64 or secure_state_dict[key].dtype == torch.long:
                    mean_tensor = mean_tensor.long()
                
                # Thêm nhiễu ngẫu nhiên nếu cần
                if self.secure_aggregation and mean_tensor.is_floating_point():
                    noise_scale = 0.001  # Cân bằng giữa bảo mật và chính xác
                    noise = torch.randn_like(mean_tensor) * noise_scale
                    mean_tensor += noise
                    
                secure_state_dict[key] = mean_tensor
                
        secure_model.load_state_dict(secure_state_dict)
        return secure_model
        
    def run_federated_round(self):
        """
        Thực hiện một vòng học liên kết hoàn chỉnh kết hợp với các cơ chế tự thích ứng
        để tối ưu hóa quá trình đồng bộ hóa giữa các shard
        """
        start_time = time.time()
        
        # 1. Chọn clients tham gia vòng này với cơ chế thích ứng dựa trên hiệu suất
        all_clients = list(self.clients.keys())
        
        if not all_clients:
            print("Không có clients nào để thực hiện vòng học liên kết")
            return
        
        # Sử dụng lựa chọn thích ứng: ưu tiên các client có hiệu suất tốt hơn
        if hasattr(self, "adaptive_selection") and self.adaptive_selection:
            # Tính điểm cho mỗi client dựa trên hiệu suất gần đây
            client_scores = {}
            
            for client_id in all_clients:
                client_data = self.clients[client_id]
                
                # Điểm cơ bản
                score = 1.0
                
                # Thưởng thêm cho việc cập nhật gần đây
                time_since_update = time.time() - client_data["last_update"]
                recency_score = max(0.5, min(1.5, 1.0 - (time_since_update / 3600.0)))  # Thời gian tính bằng giờ
                
                # Thưởng thêm cho hiệu suất tốt
                if "performance_history" in client_data and client_data["performance_history"]:
                    recent_perf = client_data["performance_history"][-1]
                    success_rate = recent_perf.get("success_rate", 0.5)
                    perf_score = 0.5 + success_rate  # 0.5 - 1.5
                else:
                    perf_score = 1.0
                
                # Điểm tổng hợp
                client_scores[client_id] = score * recency_score * perf_score
            
            # Chọn các client có điểm cao nhất
            sorted_clients = sorted(all_clients, key=lambda x: client_scores.get(x, 0), reverse=True)
            num_to_select = max(self.min_clients, int(len(all_clients) * self.client_fraction))
            selected_clients = sorted_clients[:num_to_select]
        else:
            # Chọn ngẫu nhiên như ban đầu
            selected_clients = self.select_clients()
        
        if not selected_clients:
            print("Không đủ clients để thực hiện vòng học liên kết")
            return
            
        print(f"\n==== Bắt đầu vòng học liên kết với {len(selected_clients)} clients ====")
        
        # 2. Thu thập các mô hình cục bộ từ các client được chọn
        print(f"Thu thập mô hình từ {len(selected_clients)} clients: {selected_clients}")
        
        # 3. Tổng hợp các mô hình với cơ chế thích ứng và bảo vệ
        if self.secure_aggregation:
            # Kích hoạt cơ chế tổng hợp an toàn
            client_models = {client_id: self.clients[client_id]["model"] for client_id in selected_clients}
            
            # Thêm cơ chế weight clipping để giảm ảnh hưởng của các mô hình bất thường
            if hasattr(self, "use_weight_clipping") and self.use_weight_clipping:
                for client_id, model in client_models.items():
                    self._apply_weight_clipping(model, clip_threshold=5.0)
            
            # Thực hiện tổng hợp an toàn
            secure_aggregated_model = self.secure_aggregate(client_models)
            
            if secure_aggregated_model:
                # Lưu mô hình toàn cục trước khi cập nhật
                old_model = copy.deepcopy(self.global_model) if self.global_model else None
                
                # Cập nhật mô hình toàn cục
                self.global_model = secure_aggregated_model
                self.global_model_version += 1
                
                # Đánh giá mức độ cải thiện
                if old_model:
                    improvement = self._evaluate_model_improvement(old_model, self.global_model)
                    self.performance_benchmark["model_improvements"].append(improvement)
                    print(f"Mô hình toàn cục cải thiện: {improvement:.4f}")
        else:
            # Sử dụng cơ chế tổng hợp thông thường
            self.aggregate_models(selected_clients)
            
        # 4. Phân phối mô hình toàn cục cho các client với cơ chế cá nhân hóa
        print(f"Phân phối mô hình toàn cục v{self.global_model_version} cho các clients")
        
        # Thêm cơ chế cá nhân hóa cho từng client
        if hasattr(self, "use_personalization") and self.use_personalization:
            for client_id in all_clients:
                if client_id in self.clients:
                    # Tạo bản sao của mô hình toàn cục
                    client_model = copy.deepcopy(self.global_model)
                    
                    # Áp dụng cá nhân hóa dựa trên đặc điểm của client
                    self._apply_personalization(client_model, client_id)
                    
                    # Cập nhật mô hình cho client
                    self.clients[client_id]["model"] = client_model
                    self.clients[client_id]["last_update"] = time.time()
                    
                    # Cập nhật instance nếu có
                    if "client_instance" in self.clients[client_id]:
                        client_instance = self.clients[client_id]["client_instance"]
                        if hasattr(client_instance, "update_model_from_global"):
                            client_instance.update_model_from_global(client_model)
        else:
            # Phân phối mô hình toàn cục như ban đầu
            self.distribute_global_model()
        
        # Thống kê thời gian và tài nguyên
        elapsed_time = time.time() - start_time
        self.performance_benchmark["aggregation_time"].append(elapsed_time)
        
        # Ước tính overhead của giao tiếp
        if len(selected_clients) > 0:
            # Ước tính kích thước mô hình (bytes)
            if hasattr(self.global_model, "state_dict"):
                model_size_bytes = sum(param.nelement() * param.element_size() 
                                   for param in self.global_model.state_dict().values())
                
                # Tính tổng overhead giao tiếp
                comm_overhead = model_size_bytes * len(selected_clients) * 2  # upload + download
                self.performance_benchmark["communication_overhead"].append(comm_overhead)
                
                print(f"Overhead giao tiếp: {comm_overhead / (1024*1024):.2f} MB")
        
        print(f"Vòng học liên kết hoàn thành trong {elapsed_time:.2f} giây")
        print(f"==== Kết thúc vòng học liên kết ====\n")
        
    def _apply_weight_clipping(self, model, clip_threshold=5.0):
        """
        Áp dụng clipping trọng số để giảm ảnh hưởng của các mô hình bất thường
        
        Args:
            model: Mô hình cần áp dụng clipping
            clip_threshold: Ngưỡng clipping
        """
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
            
            for key, param in state_dict.items():
                if param.dtype == torch.float32 or param.dtype == torch.float64:
                    # Chỉ áp dụng clipping cho các tham số kiểu float
                    torch.clamp_(param, min=-clip_threshold, max=clip_threshold)
            
            model.load_state_dict(state_dict)
    
    def _apply_personalization(self, model, client_id):
        """
        Áp dụng cá nhân hóa cho mô hình dựa trên đặc điểm của client
        
        Args:
            model: Mô hình cần cá nhân hóa
            client_id: ID của client
        """
        if not hasattr(model, "state_dict") or client_id not in self.clients:
            return
            
        client_data = self.clients[client_id]
        
        # Nếu client có mô hình cũ và dữ liệu hiệu suất
        if "model" in client_data and "performance_history" in client_data:
            old_model = client_data["model"]
            
            if not hasattr(old_model, "state_dict"):
                return
                
            # Lấy các tham số của mô hình cũ và mô hình mới
            old_state_dict = old_model.state_dict()
            new_state_dict = model.state_dict()
            
            # Tính hiệu suất gần đây
            recent_perf = None
            if client_data["performance_history"]:
                recent_perf = client_data["performance_history"][-1]
                
            # Hệ số cá nhân hóa dựa trên hiệu suất
            alpha = 0.9  # Mặc định 90% từ global, 10% từ local
            
            if recent_perf:
                # Điều chỉnh alpha dựa trên hiệu suất
                if "success_rate" in recent_perf:
                    success_rate = recent_perf["success_rate"]
                    # Nếu hiệu suất cao, giữ nhiều tham số cục bộ hơn
                    if success_rate > 0.7:
                        alpha = 0.7  # 70% global, 30% local
                    elif success_rate < 0.3:
                        alpha = 0.95  # 95% global, 5% local
            
            # Pha trộn các tham số
            for key in new_state_dict:
                if key in old_state_dict and new_state_dict[key].shape == old_state_dict[key].shape:
                    if new_state_dict[key].dtype == torch.float32 or new_state_dict[key].dtype == torch.float64:
                        # Pha trộn các tham số
                        new_state_dict[key] = alpha * new_state_dict[key] + (1 - alpha) * old_state_dict[key]
            
            # Cập nhật mô hình
            model.load_state_dict(new_state_dict)
    
    def _evaluate_model_improvement(self, old_model, new_model):
        """
        Đánh giá mức độ cải thiện của mô hình mới so với mô hình cũ
        
        Args:
            old_model: Mô hình cũ
            new_model: Mô hình mới
            
        Returns:
            float: Mức độ cải thiện (số dương = cải thiện, số âm = giảm chất lượng)
        """
        if not hasattr(old_model, "state_dict") or not hasattr(new_model, "state_dict"):
            return 0.0
            
        old_state_dict = old_model.state_dict()
        new_state_dict = new_model.state_dict()
        
        # Tính khoảng cách Euclidean giữa hai mô hình
        total_distance = 0.0
        param_count = 0
        
        for key in old_state_dict:
            if key in new_state_dict and old_state_dict[key].shape == new_state_dict[key].shape:
                if old_state_dict[key].dtype == torch.float32 or old_state_dict[key].dtype == torch.float64:
                    # Tính khoảng cách Euclidean bình phương
                    dist = torch.norm(new_state_dict[key] - old_state_dict[key])
                    total_distance += dist.item()
                    param_count += old_state_dict[key].nelement()
        
        # Chuẩn hóa khoảng cách
        if param_count > 0:
            return total_distance / param_count
        else:
            return 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về quá trình học liên kết
        
        Returns:
            Từ điển thống kê
        """
        stats = {
            "total_clients": len(self.clients),
            "global_model_version": self.global_model_version,
            "aggregation_time_avg": np.mean(self.performance_benchmark["aggregation_time"]) if self.performance_benchmark["aggregation_time"] else 0,
            "client_participation": {},
            "client_performance": {}
        }
        
        # Thêm thông tin tham gia của client
        for client_id, client_data in self.clients.items():
            stats["client_participation"][client_id] = {
                "updates_count": client_data["updates_count"],
                "last_update": client_data["last_update"]
            }
            
            # Thêm chỉ số hiệu suất gần đây
            if "performance_history" in client_data and client_data["performance_history"]:
                recent_performance = client_data["performance_history"][-5:]  # 5 cập nhật gần nhất
                avg_performance = {}
                
                # Tính trung bình cho mỗi chỉ số
                for entry in recent_performance:
                    for key, value in entry.items():
                        if key not in avg_performance:
                            avg_performance[key] = []
                        avg_performance[key].append(value)
                
                # Tính trung bình
                for key, values in avg_performance.items():
                    if values:
                        avg_performance[key] = sum(values) / len(values)
                
                stats["client_performance"][client_id] = avg_performance
                
        return stats
        
    def save_model(self, path: str):
        """
        Lưu mô hình toàn cục
        
        Args:
            path: Đường dẫn để lưu mô hình
        """
        if self.global_model:
            torch.save({
                "model_state_dict": self.global_model.state_dict(),
                "version": self.global_model_version,
                "timestamp": time.time()
            }, path)
            print(f"Đã lưu mô hình toàn cục v{self.global_model_version} tại {path}")
            
    def load_model(self, path: str):
        """
        Tải mô hình toàn cục
        
        Args:
            path: Đường dẫn để tải mô hình
        """
        if not self.global_model:
            print("Không thể tải mô hình toàn cục vì chưa được khởi tạo")
            return
            
        checkpoint = torch.load(path)
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.global_model_version = checkpoint.get("version", self.global_model_version + 1)
        print(f"Đã tải mô hình toàn cục v{self.global_model_version} từ {path}") 