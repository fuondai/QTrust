import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import copy

class FederatedModel(nn.Module):
    """
    Mô hình cơ sở cho Federated Learning.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Khởi tạo mô hình với các tham số cơ bản.
        
        Args:
            input_size: Kích thước đầu vào
            hidden_size: Kích thước lớp ẩn
            output_size: Kích thước đầu ra
        """
        super(FederatedModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        """
        Truyền dữ liệu qua mô hình.
        
        Args:
            x: Dữ liệu đầu vào
            
        Returns:
            Dữ liệu đầu ra từ mô hình
        """
        return self.layers(x)

class FederatedClient:
    """
    Đại diện cho một client tham gia vào quá trình Federated Learning.
    """
    def __init__(self, 
                 client_id: int, 
                 model: nn.Module, 
                 optimizer_class: torch.optim.Optimizer = optim.Adam,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 trust_score: float = 0.7,
                 device: str = 'cpu'):
        """
        Khởi tạo client cho Federated Learning.
        
        Args:
            client_id: ID duy nhất của client
            model: Mô hình cần học
            optimizer_class: Lớp optimizer sử dụng cho quá trình học
            learning_rate: Tốc độ học
            local_epochs: Số epoch huấn luyện cục bộ
            batch_size: Kích thước batch
            trust_score: Điểm tin cậy của client
            device: Thiết bị sử dụng (CPU hoặc GPU)
        """
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.trust_score = trust_score
        self.device = device
        
        # Lịch sử huấn luyện
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Dữ liệu cục bộ
        self.local_train_data = None
        self.local_val_data = None
    
    def set_local_data(self, train_data: Tuple[torch.Tensor, torch.Tensor], 
                       val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Thiết lập dữ liệu cục bộ cho client.
        
        Args:
            train_data: Tuple (features, labels) cho tập huấn luyện
            val_data: Tuple (features, labels) cho tập validation
        """
        self.local_train_data = train_data
        self.local_val_data = val_data
    
    def train_local_model(self, loss_fn: Callable = nn.MSELoss()):
        """
        Huấn luyện mô hình cục bộ với dữ liệu của client.
        
        Args:
            loss_fn: Hàm mất mát sử dụng cho huấn luyện
            
        Returns:
            Dict: Dictionary chứa lịch sử mất mát và số mẫu được huấn luyện
        """
        if self.local_train_data is None:
            raise ValueError("Client không có dữ liệu cục bộ để huấn luyện!")
        
        x_train, y_train = self.local_train_data
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)
        
        dataset_size = len(x_train)
        indices = list(range(dataset_size))
        
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            # Shuffle dữ liệu ở mỗi epoch
            np.random.shuffle(indices)
            
            running_loss = 0.0
            batches = 0
            
            # Huấn luyện theo batch
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_x = x_train[batch_indices]
                batch_y = y_train[batch_indices]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = loss_fn(outputs, batch_y)
                
                # Backward pass và tối ưu hóa
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                batches += 1
            
            epoch_loss = running_loss / batches if batches > 0 else 0
            epoch_losses.append(epoch_loss)
            self.train_loss_history.append(epoch_loss)
        
        # Tính toán mất mát trên tập validation nếu có
        val_loss = None
        if self.local_val_data is not None:
            val_loss = self.evaluate()
            self.val_loss_history.append(val_loss)
        
        return {
            'client_id': self.client_id,
            'train_loss': epoch_losses,
            'val_loss': val_loss,
            'samples': dataset_size,
            'trust_score': self.trust_score
        }
    
    def evaluate(self, loss_fn: Callable = nn.MSELoss()):
        """
        Đánh giá mô hình trên tập validation cục bộ.
        
        Args:
            loss_fn: Hàm mất mát sử dụng cho đánh giá
            
        Returns:
            float: Mất mát trên tập validation
        """
        if self.local_val_data is None:
            return None
        
        x_val, y_val = self.local_val_data
        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_val)
            loss = loss_fn(outputs, y_val).item()
        
        return loss
    
    def get_model_params(self):
        """
        Lấy tham số của mô hình cục bộ.
        
        Returns:
            OrderedDict: Tham số của mô hình
        """
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_params(self, params):
        """
        Cập nhật tham số của mô hình cục bộ.
        
        Args:
            params: Tham số mới cho mô hình
        """
        self.model.load_state_dict(copy.deepcopy(params))

class FederatedLearning:
    """
    Quản lý quá trình học liên hợp (Federated Learning).
    """
    def __init__(self, 
                 global_model: nn.Module,
                 aggregation_method: str = 'fedavg',
                 client_selection_method: str = 'random',
                 min_clients_per_round: int = 3,
                 trust_threshold: float = 0.5,
                 device: str = 'cpu'):
        """
        Khởi tạo trình quản lý Federated Learning.
        
        Args:
            global_model: Mô hình toàn cục ban đầu
            aggregation_method: Phương pháp tổng hợp ('fedavg', 'fedprox', 'fedtrust')
            client_selection_method: Phương pháp chọn client ('random', 'trust_based')
            min_clients_per_round: Số lượng tối thiểu client mỗi vòng
            trust_threshold: Ngưỡng tin cậy để chọn client
            device: Thiết bị sử dụng (CPU hoặc GPU)
        """
        self.global_model = global_model.to(device)
        self.clients = {}  # client_id -> FederatedClient
        self.aggregation_method = aggregation_method
        self.client_selection_method = client_selection_method
        self.min_clients_per_round = min_clients_per_round
        self.trust_threshold = trust_threshold
        self.device = device
        
        # Lịch sử huấn luyện toàn cục
        self.global_train_loss = []
        self.global_val_loss = []
        self.round_metrics = []
        
        # Thông số về việc không bị giám sát
        self.mu = 0.01  # Hằng số điều chỉnh cho FedProx
    
    def add_client(self, client: FederatedClient):
        """
        Thêm một client mới vào hệ thống.
        
        Args:
            client: Đối tượng FederatedClient cần thêm
        """
        self.clients[client.client_id] = client
    
    def select_clients(self, round_num: int, fraction: float = 0.5) -> List[int]:
        """
        Chọn các client tham gia vào vòng học hiện tại.
        
        Args:
            round_num: Số hiệu vòng hiện tại
            fraction: Tỷ lệ client được chọn
            
        Returns:
            List[int]: Danh sách ID của các client được chọn
        """
        eligible_clients = [
            cid for cid, client in self.clients.items() 
            if client.trust_score >= self.trust_threshold
        ]
        
        if not eligible_clients:
            eligible_clients = list(self.clients.keys())
        
        num_clients = max(self.min_clients_per_round, int(fraction * len(eligible_clients)))
        num_clients = min(num_clients, len(eligible_clients))
        
        if self.client_selection_method == 'random':
            # Chọn ngẫu nhiên
            selected_clients = np.random.choice(eligible_clients, num_clients, replace=False).tolist()
        elif self.client_selection_method == 'trust_based':
            # Chọn dựa trên điểm tin cậy
            client_trust = [(cid, self.clients[cid].trust_score) for cid in eligible_clients]
            client_trust.sort(key=lambda x: x[1], reverse=True)
            selected_clients = [cid for cid, _ in client_trust[:num_clients]]
        else:
            raise ValueError(f"Phương pháp chọn client không hợp lệ: {self.client_selection_method}")
        
        return selected_clients
    
    def train_round(self, 
                   round_num: int, 
                   client_fraction: float = 0.5, 
                   loss_fn: Callable = nn.MSELoss(),
                   global_val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Thực hiện một vòng huấn luyện Federated Learning.
        
        Args:
            round_num: Số hiệu vòng hiện tại
            client_fraction: Tỷ lệ client tham gia
            loss_fn: Hàm mất mát sử dụng
            global_val_data: Dữ liệu validation toàn cục
            
        Returns:
            Dict: Tổng hợp các chỉ số đánh giá trong vòng
        """
        selected_clients = self.select_clients(round_num, client_fraction)
        
        if len(selected_clients) == 0:
            print("Không có client nào được chọn trong vòng này!")
            return None
        
        print(f"Vòng {round_num}: Đã chọn {len(selected_clients)} clients")
        
        # Gửi mô hình toàn cục đến các client được chọn
        global_params = self.global_model.state_dict()
        for client_id in selected_clients:
            self.clients[client_id].set_model_params(global_params)
        
        # Huấn luyện cục bộ trên các client được chọn
        client_updates = {}
        for client_id in selected_clients:
            print(f"Huấn luyện client {client_id}...")
            client_result = self.clients[client_id].train_local_model(loss_fn)
            client_updates[client_id] = {
                'params': self.clients[client_id].get_model_params(),
                'metrics': client_result
            }
        
        # Tổng hợp các cập nhật
        self.aggregate_updates(client_updates)
        
        # Đánh giá mô hình toàn cục trên tập validation
        val_loss = None
        if global_val_data is not None:
            x_val, y_val = global_val_data
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            
            self.global_model.eval()
            with torch.no_grad():
                outputs = self.global_model(x_val)
                val_loss = loss_fn(outputs, y_val).item()
            
            self.global_val_loss.append(val_loss)
        
        # Tính toán mất mát trung bình của client
        avg_train_loss = np.mean([
            np.mean(update['metrics']['train_loss']) 
            for _, update in client_updates.items()
        ])
        self.global_train_loss.append(avg_train_loss)
        
        # Lưu các chỉ số đánh giá cho vòng hiện tại
        round_metrics = {
            'round': round_num,
            'clients': selected_clients,
            'avg_train_loss': avg_train_loss,
            'val_loss': val_loss,
            'client_metrics': {
                cid: update['metrics'] for cid, update in client_updates.items()
            }
        }
        
        self.round_metrics.append(round_metrics)
        return round_metrics
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật từ các client.
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        if not client_updates:
            return
        
        if self.aggregation_method == 'fedavg':
            self._aggregate_fedavg(client_updates)
        elif self.aggregation_method == 'fedtrust':
            self._aggregate_fedtrust(client_updates)
        else:
            raise ValueError(f"Phương pháp tổng hợp không hợp lệ: {self.aggregation_method}")
    
    def _aggregate_fedavg(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật theo thuật toán FedAvg.
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        # Lấy trọng số dựa trên số lượng mẫu từ mỗi client
        total_samples = sum(update['metrics']['samples'] for _, update in client_updates.items())
        
        # Khởi tạo tham số tổng hợp
        global_params = self.global_model.state_dict()
        
        # Tính tổng tham số có trọng số
        for client_id, update in client_updates.items():
            client_params = update['params']
            weight = update['metrics']['samples'] / total_samples
            
            for key in global_params.keys():
                if client_id == list(client_updates.keys())[0]:
                    # Nếu là client đầu tiên, khởi tạo tham số với 0
                    global_params[key] = torch.zeros_like(global_params[key])
                
                # Thêm tham số của client với trọng số
                global_params[key] += client_params[key] * weight
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
    
    def _aggregate_fedtrust(self, client_updates: Dict[int, Dict[str, Any]]):
        """
        Tổng hợp cập nhật theo thuật toán FedTrust (tính đến mức độ tin cậy).
        
        Args:
            client_updates: Dictionary ánh xạ client_id thành dict chứa params và metrics
        """
        # Tính trọng số dựa trên số lượng mẫu và điểm tin cậy
        total_weight = sum(
            update['metrics']['samples'] * update['metrics']['trust_score']
            for _, update in client_updates.items()
        )
        
        if total_weight == 0:
            # Fallback về FedAvg nếu tổng trọng số là 0
            return self._aggregate_fedavg(client_updates)
        
        # Khởi tạo tham số tổng hợp
        global_params = self.global_model.state_dict()
        
        # Tính tổng tham số có trọng số
        for client_id, update in client_updates.items():
            client_params = update['params']
            weight = (update['metrics']['samples'] * update['metrics']['trust_score']) / total_weight
            
            for key in global_params.keys():
                if client_id == list(client_updates.keys())[0]:
                    # Nếu là client đầu tiên, khởi tạo tham số với 0
                    global_params[key] = torch.zeros_like(global_params[key])
                
                # Thêm tham số của client với trọng số
                global_params[key] += client_params[key] * weight
        
        # Cập nhật mô hình toàn cục
        self.global_model.load_state_dict(global_params)
    
    def get_global_model(self):
        """
        Lấy mô hình toàn cục hiện tại.
        
        Returns:
            nn.Module: Mô hình toàn cục
        """
        return self.global_model
    
    def update_client_trust(self, client_id: int, trust_score: float):
        """
        Cập nhật điểm tin cậy cho một client.
        
        Args:
            client_id: ID của client
            trust_score: Điểm tin cậy mới
        """
        if client_id in self.clients:
            self.clients[client_id].trust_score = trust_score
    
    def save_global_model(self, path: str):
        """
        Lưu mô hình toàn cục vào file.
        
        Args:
            path: Đường dẫn để lưu mô hình
        """
        torch.save(self.global_model.state_dict(), path)
    
    def load_global_model(self, path: str):
        """
        Tải mô hình toàn cục từ file.
        
        Args:
            path: Đường dẫn đến file mô hình
        """
        self.global_model.load_state_dict(torch.load(path, map_location=self.device))

    def train(self, 
             num_rounds: int,
             client_fraction: float = 0.5,
             loss_fn: Callable = nn.MSELoss(),
             global_val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
             save_path: Optional[str] = None,
             early_stopping_rounds: int = 10,
             early_stopping_tolerance: float = 0.001):
        """
        Thực hiện quá trình huấn luyện Federated Learning qua nhiều vòng.
        
        Args:
            num_rounds: Số vòng huấn luyện
            client_fraction: Tỷ lệ client tham gia mỗi vòng
            loss_fn: Hàm mất mát
            global_val_data: Dữ liệu validation toàn cục
            save_path: Đường dẫn để lưu mô hình tốt nhất
            early_stopping_rounds: Số vòng chờ trước khi dừng sớm
            early_stopping_tolerance: Ngưỡng cải thiện tối thiểu
            
        Returns:
            Dict: Dictionary chứa lịch sử huấn luyện
        """
        best_val_loss = float('inf')
        rounds_without_improvement = 0
        
        for round_num in range(1, num_rounds + 1):
            round_metrics = self.train_round(
                round_num, client_fraction, loss_fn, global_val_data
            )
            
            # In kết quả
            print(f"Vòng {round_num}: "
                 f"Mất mát huấn luyện = {round_metrics['avg_train_loss']:.4f}, "
                 f"Mất mát validation = {round_metrics['val_loss']:.4f if round_metrics['val_loss'] is not None else 'N/A'}")
            
            # Kiểm tra điều kiện dừng sớm
            if global_val_data is not None and round_metrics['val_loss'] is not None:
                if round_metrics['val_loss'] < best_val_loss - early_stopping_tolerance:
                    best_val_loss = round_metrics['val_loss']
                    rounds_without_improvement = 0
                    
                    # Lưu mô hình tốt nhất
                    if save_path:
                        self.save_global_model(save_path)
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= early_stopping_rounds:
                    print(f"Dừng sớm tại vòng {round_num} do không có cải thiện sau {early_stopping_rounds} vòng")
                    break
        
        # Tải mô hình tốt nhất nếu đã lưu
        if save_path and global_val_data is not None:
            self.load_global_model(save_path)
        
        # Trả về lịch sử huấn luyện
        return {
            'train_loss': self.global_train_loss,
            'val_loss': self.global_val_loss,
            'round_metrics': self.round_metrics
        } 