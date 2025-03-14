"""
MAD-RAPID: Multi-Agent Dynamic - Routing with Adaptive Prioritization in Distributed Shards
Module này triển khai giao thức truyền thông xuyên mảnh tiên tiến sử dụng dự đoán
tắc nghẽn và học sâu để tối ưu hóa đường dẫn xuyên mảnh.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
import random
import time
from collections import defaultdict, deque
from dqn_blockchain_sim.blockchain.network import BlockchainNetwork
from dqn_blockchain_sim.blockchain.transaction import TransactionStatus
import torch.optim as optim
import math
import heapq

class LSTMCongestionPredictor(nn.Module):
    """
    Mô hình LSTM để dự đoán mức độ tắc nghẽn tương lai của các shard
    """
    
    def __init__(self, input_size: int = 8, hidden_size: int = 128, num_layers: int = 2, output_size: int = 1):
        """
        Khởi tạo mô hình dự đoán tắc nghẽn
        
        Args:
            input_size: Số chiều của vector đầu vào (các đặc trưng của shard)
            hidden_size: Số đơn vị ẩn trong LSTM
            num_layers: Số lớp LSTM xếp chồng
            output_size: Số chiều đầu ra (mặc định là 1 - dự đoán mức độ tắc nghẽn)
        """
        super(LSTMCongestionPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Lớp fully-connected đầu ra
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, hidden=None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass của mô hình LSTM
        
        Args:
            x: Tensor đầu vào chứa các đặc trưng của shard trong nhiều bước thời gian
               [batch_size, seq_len, input_size]
            hidden: Trạng thái ẩn ban đầu của LSTM (tùy chọn)
            
        Returns:
            outputs: Tensor đầu ra chứa dự đoán tắc nghẽn
            hidden: Trạng thái ẩn cuối cùng của LSTM
        """
        # Kiểm tra kích thước đầu vào
        if x.size(-1) != self.input_size:
            # Nếu kích thước không đúng, cắt bớt hoặc pad thêm
            if x.size(-1) > self.input_size:
                # Cắt bớt nếu quá lớn
                x = x[..., :self.input_size]
            else:
                # Thêm padding nếu quá nhỏ
                padding = torch.zeros(*x.shape[:-1], self.input_size - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
        
        # LSTM layer
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Fully connected layer
        outputs = self.fc(lstm_out)
        
        return outputs, hidden
        
    def predict(self, feature_history):
        """
        Dự đoán mức độ tắc nghẽn từ lịch sử đặc trưng
        
        Args:
            feature_history: Mảng numpy chứa các vector đặc trưng trong quá khứ
                            shape (seq_len, input_size)
                            
        Returns:
            float: Mức độ tắc nghẽn dự đoán (0-1)
        """
        # Chuyển đổi thành tensor
        if isinstance(feature_history, np.ndarray):
            # Đảm bảo kích thước đúng [batch_size, seq_len, input_size]
            if len(feature_history.shape) == 2:
                feature_history = feature_history.reshape(1, *feature_history.shape)
            x = torch.FloatTensor(feature_history)
        else:
            # Đã là tensor, đảm bảo kích thước đúng
            if len(feature_history.shape) == 2:
                feature_history = feature_history.unsqueeze(0)
            x = feature_history
        
        # Đặt mạng ở chế độ đánh giá
        self.eval()
        
        # Dự đoán không tính gradient
        with torch.no_grad():
            outputs, _ = self.forward(x)
            # Lấy dự đoán cuối cùng và chuyển về giá trị scalar
            prediction = outputs[0, -1, 0].item()
            # Đảm bảo giá trị nằm trong khoảng [0, 1]
            prediction = max(0.0, min(1.0, prediction))
            
        return prediction


class AttentionBasedPathOptimizer:
    """
    Bộ tối ưu hóa đường dẫn dựa trên cơ chế chú ý (attention)
    """
    
    def __init__(self, embedding_dim: int = 64, attention_heads: int = 4):
        """
        Khởi tạo bộ tối ưu hóa đường dẫn
        
        Args:
            embedding_dim: Kích thước của vector nhúng
            attention_heads: Số đầu chú ý
        """
        self.embedding_dim = embedding_dim
        self.attention_heads = attention_heads
        
        # Mô hình chú ý đa đầu
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_heads,
            batch_first=True
        )
        
    def optimize(self, 
               source_embedding: torch.Tensor,
               target_embedding: torch.Tensor,
               all_embeddings: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
        """
        Tối ưu hóa đường dẫn dựa trên vector nhúng của các shard
        
        Args:
            source_embedding: Vector nhúng của shard nguồn
            target_embedding: Vector nhúng của shard đích
            all_embeddings: Ma trận chứa vector nhúng của tất cả các shard
            
        Returns:
            Tuple[List[int], torch.Tensor]: Đường dẫn tối ưu và trọng số chú ý
        """
        with torch.no_grad():  # Không tính toán gradient trong quá trình này
            # Đảm bảo các tensor có kích thước đúng cho multihead_attention
            # Tensor query cần có kích thước [batch_size, seq_len, embed_dim]
            source_expanded = source_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
            all_embeddings_expanded = all_embeddings.unsqueeze(0)  # [1, num_shards, embedding_dim]
            
            # Tính toán attention scores sử dụng multihead_attention
            attn_output, attention_weights = self.multihead_attention(
                source_expanded,  # query: [1, 1, embedding_dim]
                all_embeddings_expanded,  # key: [1, num_shards, embedding_dim]
                all_embeddings_expanded   # value: [1, num_shards, embedding_dim]
            )
            
            # Chọn đường dẫn dựa trên điểm chú ý
            num_shards = all_embeddings.shape[0]
            attention_weights = attention_weights.squeeze(0)  # [1, num_shards]
            
            # Đơn giản hóa: Chọn các shard có điểm cao nhất
            _, indices = torch.topk(attention_weights, min(3, num_shards))
            path = indices.squeeze(0).tolist()
            
            # Đảm bảo shard nguồn và đích có trong đường dẫn
            source_idx = 0  # Giả định shard nguồn là 0
            target_idx = num_shards - 1  # Giả định shard đích là shard cuối cùng
            
            if source_idx not in path:
                path.insert(0, source_idx)
            if target_idx not in path:
                path.append(target_idx)
            
            return path, attention_weights
    
    def find_optimal_path(self, source_embedding, all_embeddings, congestion_tensor, latency_tensor):
        """
        Tìm đường dẫn tối ưu từ shard nguồn đến shard đích.
        
        Args:
            source_embedding: Vector nhúng của shard nguồn
            all_embeddings: Ma trận chứa vector nhúng của tất cả các shard
            congestion_tensor: Tensor chứa dự đoán tắc nghẽn của các shard
            latency_tensor: Ma trận độ trễ giữa các cặp shard
            
        Returns:
            List[int]: Đường dẫn tối ưu (danh sách các shard IDs)
        """
        with torch.no_grad():  # Không tính toán gradient trong quá trình này
            # Đảm bảo các tensor có kích thước đúng cho multihead_attention
            # Tensor query cần có kích thước [batch_size, seq_len, embed_dim]
            source_expanded = source_embedding.unsqueeze(0).unsqueeze(0)  # [1, 1, embedding_dim]
            all_embeddings_expanded = all_embeddings.unsqueeze(0)  # [1, num_shards, embedding_dim]
            
            # Tính toán attention scores sử dụng multihead_attention
            attn_output, attention_weights = self.multihead_attention(
                source_expanded,  # query: [1, 1, embedding_dim]
                all_embeddings_expanded,  # key: [1, num_shards, embedding_dim]
                all_embeddings_expanded   # value: [1, num_shards, embedding_dim]
            )
            
            # Điều chỉnh scores dựa trên dự đoán tắc nghẽn
            attention_weights = attention_weights.squeeze(0)  # [1, num_shards]
            
            # Đảm bảo congestion_tensor có kích thước đúng
            if congestion_tensor.dim() == 1:
                congestion_tensor = congestion_tensor.unsqueeze(0)  # [1, num_shards]
            
            # Điều chỉnh theo congestion
            adjusted_scores = attention_weights * (1.0 - congestion_tensor)
            
            # Đảm bảo latency_tensor có kích thước đúng
            num_shards = all_embeddings.shape[0]
            
            # Giảm congestion_tensor và adjusted_scores về 1D nếu cần
            if adjusted_scores.dim() > 1:
                adjusted_scores = adjusted_scores.squeeze(0)  # [num_shards]
            
            # Tạo ma trận điểm số kết hợp
            # Mỗi phần tử (i,j) trong ma trận này đại diện cho điểm số của
            # việc đi từ nút i đến nút j, dựa trên attention và congestion
            combined_scores = torch.ones_like(latency_tensor)
            for i in range(num_shards):
                for j in range(num_shards):
                    if i != j:
                        # Kết hợp điểm chú ý với ma trận độ trễ
                        combined_scores[i, j] = adjusted_scores[j] / latency_tensor[i, j]
                    else:
                        combined_scores[i, j] = 1.0  # Tự liên kết có điểm cao
            
            # Sử dụng thuật toán Dijkstra để tìm đường dẫn tối ưu
            source_shard_id = 0  # Giả định shard nguồn là 0
            target_shard_id = num_shards - 1  # Giả định shard đích là shard cuối cùng
            
            path = self._dijkstra_search(
                combined_scores.detach().cpu().numpy(),
                latency_tensor.detach().cpu().numpy(),
                source_shard_id, 
                target_shard_id
            )
            
            return path

    def _dijkstra_search(self, 
                      attention_scores: np.ndarray,
                      latency_matrix: np.ndarray,
                      source: int, 
                      target: int) -> List[int]:
        """
        Thuật toán Dijkstra có điều chỉnh theo trọng số chú ý
        
        Args:
            attention_scores: Ma trận điểm số kết hợp giữa attention và độ trễ
            latency_matrix: Ma trận độ trễ giữa các shard
            source: Shard nguồn
            target: Shard đích
            
        Returns:
            Đường dẫn tối ưu (danh sách các shard)
        """
        num_nodes = latency_matrix.shape[0]
        
        # Khởi tạo khoảng cách và đường dẫn
        distance = np.ones(num_nodes) * np.inf
        distance[source] = 0
        visited = [False] * num_nodes
        parent = [-1] * num_nodes
        
        for _ in range(num_nodes):
            # Tìm đỉnh không ghé thăm có khoảng cách nhỏ nhất
            min_dist = np.inf
            min_node = -1
            
            for node in range(num_nodes):
                if not visited[node] and distance[node] < min_dist:
                    min_dist = distance[node]
                    min_node = node
                    
            if min_node == -1:
                break
            
            # Đánh dấu đỉnh đã ghé thăm
            visited[min_node] = True
            
            # Cập nhật khoảng cách cho các đỉnh kề
            for neighbor in range(num_nodes):
                if (latency_matrix[min_node, neighbor] > 0 and 
                    not visited[neighbor]):
                    
                    # Sử dụng ma trận điểm số trực tiếp
                    edge_weight = 1.0 / max(0.1, attention_scores[min_node, neighbor])
                    
                    if distance[min_node] + edge_weight < distance[neighbor]:
                        distance[neighbor] = distance[min_node] + edge_weight
                        parent[neighbor] = min_node
        
        # Xây dựng đường dẫn từ nguồn đến đích
        path = []
        current = target
        
        # Nếu không có đường dẫn đến đích, trả về đường đi trực tiếp
        if parent[target] == -1 and source != target:
            return [source, target]
        
        # Xây dựng đường dẫn
        while current != -1:
            path.append(current)
            current = parent[current]
        
        # Đảo ngược đường dẫn
        return path[::-1]


class MADRAPIDProtocol:
    """
    Giao thức MAD-RAPID triển khai lộ trình tối ưu đa tác tử và đường dẫn ưu tiên
    thích ứng cho giao dịch xuyên mảnh
    """
    
    def __init__(self, network, config=None):
        """
        Khởi tạo MAD-RAPID protocol
        
        Args:
            network: Tham chiếu đến mạng blockchain
            config: Dictionary chứa cấu hình
        """
        self.network = network
        self.config = config if config is not None else {}
        self.use_dqn = self.config.get("use_dqn", True)
        
        # Cấu hình federated learning
        self.use_federated_learning = self.config.get("use_federated_learning", False)
        self.federated_learning = None
        
        self.stats = {
            "optimized_tx_count": 0,
            "total_tx_processed": 0,
            "path_selection_stats": {
                "optimized": 0,
                "direct": 0,
                "direct_after_fail": 0,
                "rejected": 0
            },
            "latency_improvement": 0.0,
            "energy_saved": 0.0,
            "processing_times": []
        }
        
        # Thêm các biến thống kê mới
        self.total_tx_count = 0          # Tổng số giao dịch đã xử lý
        self.successful_tx_count = 0     # Tổng số giao dịch thành công
        self.optimized_tx_count = 0      # Tổng số giao dịch được tối ưu hóa
        
        # Lưu thời gian bắt đầu để tính toán thống kê
        self.start_time = time.time()
        
        # Khởi tạo LSTM cho dự đoán tắc nghẽn
        self.congestion_predictor = None
        if self.config.get("use_lstm_predictor", True):
            try:
                self.congestion_predictor = LSTMCongestionPredictor(
                    input_size=self.config.get("lstm_input_size", 8),
                    hidden_size=self.config.get("lstm_hidden_size", 128),
                    num_layers=self.config.get("lstm_num_layers", 2),
                    output_size=1
                )
            except Exception as e:
                print(f"Lỗi khi khởi tạo LSTM Congestion Predictor: {e}")
        
        # Khởi tạo AttentionBasedPathOptimizer
        self.path_optimizer = None
        if self.config.get("use_attention_optimizer", True):
            try:
                self.path_optimizer = AttentionBasedPathOptimizer(
                    embedding_dim=self.config.get("embedding_dim", 64),
                    attention_heads=self.config.get("attention_heads", 4)
                )
            except Exception as e:
                print(f"Lỗi khi khởi tạo AttentionBasedPathOptimizer: {e}")
        
        # Thu thập feature history để dự đoán tắc nghẽn
        self.feature_history = {}
        self.shard_embeddings = None
        self.latency_matrix = None
        self.congestion_tensor = None
        
        # Khởi tạo Federated Learning nếu được kích hoạt
        if self.use_federated_learning:
            try:
                from dqn_blockchain_sim.federated.federated_learning import FederatedLearning
                self.federated_learning = FederatedLearning(
                    global_rounds=self.config.get("fl_rounds", 5),
                    local_epochs=self.config.get("fl_local_epochs", 2),
                    client_fraction=self.config.get("fl_client_fraction", 0.8),
                    aggregation_method=self.config.get("fl_aggregation_method", "fedavg"),
                    secure_aggregation=self.config.get("fl_secure_aggregation", True),
                    log_dir=self.config.get("log_dir", "logs")
                )
                print("Đã khởi tạo Federated Learning trong MAD-RAPID")
            except ImportError as e:
                print(f"Không thể tải module Federated Learning: {e}")
                self.use_federated_learning = False
        
        # Khởi tạo DQN agents
        self.dqn_agents = {}
        if self.use_dqn:
            self.initialize_dqn_agents()

    def initialize_dqn_agents(self):
        """
        Khởi tạo các DQN agent cho mỗi shard
        """
        from dqn_blockchain_sim.agents.dqn_agent import ShardDQNAgent
        
        print(f"Khởi tạo {len(self.network.shards)} DQN agents...")
        for shard_id, shard in self.network.shards.items():
            # Khởi tạo đặc trưng đầu vào và đầu ra
            state_size = self.config.get("dqn_state_size", 12)  # Kích thước vector trạng thái
            action_size = self.config.get("dqn_action_size", 3)  # Số hành động có thể
            
            # Tạo cấu hình DQN
            dqn_config = {
                "hidden_layers": [64, 128, 64],
                "learning_rate": self.config.get("dqn_learning_rate", 0.001),
                "gamma": self.config.get("dqn_gamma", 0.99),
                "epsilon": self.config.get("dqn_epsilon", 1.0),
                "epsilon_min": self.config.get("dqn_epsilon_min", 0.1),
                "epsilon_decay": self.config.get("dqn_epsilon_decay", 0.995),
                "replay_buffer_size": self.config.get("dqn_batch_size", 64) * 10,
                "target_update_freq": 10,
                "use_double_dqn": self.config.get("use_double_dqn", True),
                "use_dueling_dqn": self.config.get("use_dueling_dqn", False),
                "use_prioritized_replay": self.config.get("use_prioritized_replay", True),
                "tau": self.config.get("dqn_tau", 0.01)
            }
            
            # Tạo agent cho mỗi shard
            agent = ShardDQNAgent(
                shard_id=shard_id,
                state_size=state_size,
                action_size=action_size,
                config=dqn_config
            )
            
            self.dqn_agents[shard_id] = agent
            
        print(f"Đã khởi tạo {len(self.dqn_agents)} DQN agents")

        # Đăng ký các agent với Federated Learning nếu được bật
        if self.use_federated_learning and self.federated_learning:
            for shard_id, agent in self.dqn_agents.items():
                # Lấy kích thước dữ liệu (số lượng giao dịch trong shard)
                data_size = len(self.network.shards[shard_id].transaction_pool) if hasattr(self.network.shards[shard_id], 'transaction_pool') else 0
                
                # Đăng ký agent với federated learning
                self.federated_learning.register_client(
                    client_id=str(shard_id),
                    model=agent.policy_net,
                    data_size=data_size
                )
                
            print(f"Đã đăng ký {len(self.dqn_agents)} DQN agents với Federated Learning")

    def _train_dqn_agents(self):
        """
        Huấn luyện tất cả các agent DQN dựa trên kinh nghiệm thu thập được
        """
        if not self.use_dqn or len(self.dqn_agents) == 0:
            return
            
        # Train all DQN agents
        for shard_id, agent in self.dqn_agents.items():
            if hasattr(agent, 'train'):
                agent.train()
                
        # Tăng biến đếm tổng số bước huấn luyện
        self.total_training_steps += 1
        
        # Thực hiện cập nhật federated learning
        if self.use_federated_learning and self.federated_learning:
            for shard_id, agent in self.dqn_agents.items():
                # Cập nhật mô hình cục bộ cho federated learning
                metrics = {
                    "loss": np.mean(agent.loss_history[-10:]) if hasattr(agent, 'loss_history') and agent.loss_history else 0,
                    "reward": np.mean(agent.reward_history[-10:]) if hasattr(agent, 'reward_history') and agent.reward_history else 0,
                    "epsilon": agent.epsilon if hasattr(agent, 'epsilon') else 0
                }
                
                self.federated_learning.update_client_model(
                    client_id=str(shard_id),
                    model=agent.policy_net,
                    metrics=metrics
                )
            
            # Thực hiện một vòng federated learning sau mỗi N bước huấn luyện
            # hoặc khi đạt được một ngưỡng nhất định
            if self.total_training_steps % self.fl_rounds == 0:
                print("\nBắt đầu vòng Federated Learning...")
                self.federated_learning.run_federated_round()
                
                # Cập nhật các agent với mô hình toàn cục mới
                for shard_id, agent in self.dqn_agents.items():
                    # Cập nhật mạng chính sách từ mô hình toàn cục
                    agent.policy_net.load_state_dict(self.federated_learning.global_model.state_dict())
                    
                    # Cập nhật mạng mục tiêu nếu cần
                    if hasattr(agent, 'sync_target_network'):
                        agent.sync_target_network()
                        
                print("Đã hoàn thành vòng Federated Learning và cập nhật các agent\n")
                
                # Lưu mô hình toàn cục định kỳ
                if hasattr(self, 'log_dir'):
                    model_path = f"{self.log_dir}/federated_model_global_v{self.federated_learning.global_model_version}.pt"
                    self.federated_learning.save_model(model_path)

    def _predict_congestion(self, shard_id):
        """
        Dự đoán mức độ tắc nghẽn của một shard dựa trên dữ liệu hiện tại
        
        Args:
            shard_id: ID của shard cần dự đoán
            
        Returns:
            float: Mức độ tắc nghẽn dự đoán (0-1)
        """
        # Kiểm tra tính hợp lệ của shard_id
        if not hasattr(self, 'network') or not self.network:
            return 0.0
            
        if not hasattr(self.network, 'shards') or shard_id not in self.network.shards:
            return 0.0
            
        # Lấy thông tin cơ bản từ shard
        shard = self.network.shards[shard_id]
        
        # Dự đoán đơn giản dựa trên số lượng giao dịch
        tx_count = len(shard.transaction_pool) if hasattr(shard, 'transaction_pool') else 0
        cross_shard_tx_count = len([tx for tx in shard.transaction_pool if hasattr(tx, 'is_cross_shard') and tx.is_cross_shard]) if hasattr(shard, 'transaction_pool') else 0
        
        # Dự đoán dựa trên công thức đơn giản
        max_capacity = 100  # Giả định: mỗi shard xử lý tối đa 100 giao dịch
        base_congestion = min(1.0, tx_count / max_capacity)
        
        # Tính thêm trọng số cho giao dịch xuyên shard
        cross_shard_weight = 0.2  # Giao dịch xuyên shard tạo thêm tắc nghẽn
        cross_shard_factor = min(1.0, cross_shard_tx_count / max(1, tx_count))
        cross_shard_congestion = cross_shard_factor * cross_shard_weight
        
        # Tính tổng tắc nghẽn
        total_congestion = min(1.0, base_congestion + cross_shard_congestion)
        
        # Thêm nhiễu ngẫu nhiên nhỏ để tránh đồng nhất
        noise = random.uniform(-0.05, 0.05)
        predicted_congestion = max(0.0, min(1.0, total_congestion + noise))
        
        # Cập nhật congestion_tensor nếu đã được khởi tạo
        if hasattr(self, 'congestion_tensor') and self.congestion_tensor is not None:
            num_shards = self.congestion_tensor.shape[0]
            if shard_id < num_shards:
                # Cập nhật mức độ tắc nghẽn từ shard này đến tất cả các shard khác
                for i in range(num_shards):
                    self.congestion_tensor[shard_id, i] = predicted_congestion
                    # Tắc nghẽn khi nhận từ shard khác cũng bị ảnh hưởng
                    self.congestion_tensor[i, shard_id] = predicted_congestion * 0.8  # Giảm nhẹ khi nhận
        
        return predicted_congestion

    def get_statistics(self):
        """Lấy thống kê về hiệu suất của MAD-RAPID"""
        # Khởi tạo thống kê nếu chưa tồn tại
        if "path_selection_stats" not in self.stats:
            self.stats["path_selection_stats"] = {
                "total_attempts": 0,
                "successful_optimizations": 0,
                "direct": 0,
                "algorithm_usage": {"Dijkstra": 0},
                "total_latency": 0,
                "total_energy_saved": 0,
                "cross_shard_transactions": 0,
                "optimized_transactions": 0
            }
            
        if "total_transactions" not in self.stats:
            self.stats["total_transactions"] = 0
        if "successful_transactions" not in self.stats:
            self.stats["successful_transactions"] = 0
        if "total_cross_shard" not in self.stats:
            self.stats["total_cross_shard"] = 0
        if "successful_cross_shard" not in self.stats:
            self.stats["successful_cross_shard"] = 0
            
        # Tính toán các tỷ lệ
        total_tx = self.stats.get("total_transactions", 0)
        successful_tx = self.stats.get("successful_transactions", 0)
        total_cross = self.stats.get("total_cross_shard", 0)
        successful_cross = self.stats.get("successful_cross_shard", 0)
        
        # Lấy thống kê về độ trễ và năng lượng
        path_stats = self.stats.get("path_selection_stats", {})
        total_latency = path_stats.get("total_latency", 0)
        total_energy = path_stats.get("total_energy_saved", 0)
        total_optimizations = path_stats.get("successful_optimizations", 0)
        
        # Tính toán các chỉ số hiệu suất
        success_rate = (successful_tx / total_tx * 100) if total_tx > 0 else 0.0
        cross_shard_success_rate = (successful_cross / total_cross * 100) if total_cross > 0 else 0.0
        average_latency = total_latency / successful_tx if successful_tx > 0 else 0.0
        optimization_rate = (total_optimizations / total_cross * 100) if total_cross > 0 else 0.0
        
        return {
            "total_transactions": total_tx,
            "successful_transactions": successful_tx,
            "success_rate": success_rate,
            "cross_shard_transactions": total_cross,
            "successful_cross_shard": successful_cross,
            "cross_shard_success_rate": cross_shard_success_rate,
            "average_latency": average_latency,
            "energy_saved": total_energy,
            "optimization_rate": optimization_rate,
            "path_selection_stats": path_stats,
            "performance_metrics": self.stats.get("performance_metrics", {})
        }

    def _initialize_shard_embeddings(self):
        """
        Khởi tạo các vector embedding cho các shard.
        Vector này được sử dụng cho việc tối ưu hóa đường dẫn xuyên shard.
        """
        # Kiểm tra xem network có tồn tại không
        if not hasattr(self, 'network') or not self.network:
            print("Không thể khởi tạo shard_embeddings: network chưa được khởi tạo")
            return
            
        # Lấy số lượng shard từ network
        num_shards = len(self.network.shards) if hasattr(self.network, 'shards') else 0
        if num_shards == 0:
            print("Không thể khởi tạo shard_embeddings: không có shard nào trong network")
            return
            
        # Kích thước vector embedding
        embedding_dim = 64  # Kích thước mặc định
        
        # Khởi tạo các vector embedding ngẫu nhiên
        self.shard_embeddings = torch.randn(num_shards, embedding_dim) / math.sqrt(embedding_dim)
        
        # Tính toán ma trận độ trễ giữa các shard (đơn giản hóa)
        self.latency_matrix = torch.ones(num_shards, num_shards)
        for i in range(num_shards):
            for j in range(num_shards):
                if i != j:
                    # Độ trễ đơn giản - khoảng cách giữa các ID shard
                    self.latency_matrix[i, j] = 1.0 + abs(i - j) * 0.1
                else:
                    # Độ trễ trong cùng một shard
                    self.latency_matrix[i, j] = 0.1
        
        # Khởi tạo tensor tắc nghẽn (ban đầu đều bằng 0)
        self.congestion_tensor = torch.zeros(num_shards, dtype=torch.float32)
        
        # Khởi tạo bộ dự đoán tắc nghẽn cho mỗi shard nếu chưa có
        if not hasattr(self, 'congestion_predictors'):
            self.congestion_predictors = {}
            for shard_id in range(num_shards):
                # Mô hình LSTM cho dự đoán tắc nghẽn
                input_size = 8  # Kích thước vector đặc trưng
                hidden_size = 64  # Kích thước lớp ẩn LSTM
                num_layers = 1  # Số lớp LSTM
                self.congestion_predictors[shard_id] = LSTMCongestionPredictor(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    output_size=1
                )
        
        print(f"Đã khởi tạo embeddings cho {num_shards} shard") 

    def update_feature_history(self):
        """
        Cập nhật lịch sử đặc trưng cho các shard
        """
        # Đảm bảo network đã được khởi tạo
        if not hasattr(self, 'network') or not self.network:
            print("Không thể cập nhật feature_history: network chưa được khởi tạo")
            return
            
        # Lấy số lượng shard từ network
        num_shards = len(self.network.shards) if hasattr(self.network, 'shards') else 0
        if num_shards == 0:
            print("Không thể cập nhật feature_history: không có shard nào trong network")
            return
        
        # Khởi tạo feature_history nếu chưa có
        if not hasattr(self, 'feature_history'):
            self.feature_history = {}
            for shard_id in range(num_shards):
                self.feature_history[shard_id] = deque(maxlen=10)  # Lưu trữ 10 frame gần nhất
        
        # Khởi tạo congestion_tensor nếu chưa có
        if not hasattr(self, 'congestion_tensor') or self.congestion_tensor is None:
            self.congestion_tensor = torch.zeros(num_shards, dtype=torch.float32)
        
        # Cập nhật lịch sử đặc trưng cho mỗi shard
        for shard_id in range(num_shards):
            # Lấy shard từ network
            shard = self.network.shards.get(shard_id)
            if not shard:
                continue
                
            # Tạo vector đặc trưng đơn giản
            features = [
                len(shard.transaction_pool) if hasattr(shard, 'transaction_pool') else 0,  # Số lượng giao dịch trong pool
                len([tx for tx in shard.transaction_pool if hasattr(tx, 'is_cross_shard') and tx.is_cross_shard]) if hasattr(shard, 'transaction_pool') else 0,  # Số lượng giao dịch xuyên shard
                getattr(shard, 'congestion', 0),  # Mức độ tắc nghẽn
                getattr(shard, 'throughput', 0),  # Thông lượng
                getattr(shard, 'latency', 0),  # Độ trễ
                getattr(shard, 'block_height', 0),  # Chiều cao khối
                float(shard_id) / num_shards,  # Vị trí tương đối
                1.0  # Bias
            ]
            
            # Chuyển đổi thành numpy array và thêm vào lịch sử
            feature_vector = np.array(features, dtype=np.float32)
            
            # Đảm bảo shard_id đã được khởi tạo trong feature_history
            if shard_id not in self.feature_history:
                self.feature_history[shard_id] = deque(maxlen=10)
                
            self.feature_history[shard_id].append(feature_vector)
            
            # Cập nhật congestion_tensor
            congestion = getattr(shard, 'congestion', 0)
            self.congestion_tensor[shard_id] = torch.tensor(congestion, dtype=torch.float32)
            
        # Đảm bảo congestion_tensor nằm trong khoảng [0, 1]
        self.congestion_tensor = torch.clamp(self.congestion_tensor, 0.0, 1.0)

    def process_cross_shard_transaction(self, transaction, source_shard_id=None, target_shard_id=None):
        """
        Xử lý giao dịch chéo shard bằng cách tìm đường đi tối ưu
        
        Args:
            transaction: Giao dịch cần xử lý
            source_shard_id: ID shard nguồn (nếu không được cung cấp, sẽ sử dụng transaction.source_shard)
            target_shard_id: ID shard đích (nếu không được cung cấp, sẽ sử dụng transaction.target_shard)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Khởi tạo thống kê nếu chưa có
            if not hasattr(self, 'stats'):
                self.stats = {}
            if 'path_selection_stats' not in self.stats:
                self.stats['path_selection_stats'] = {
                    'total_attempts': 0,
                    'successful_optimizations': 0,
                    'direct_paths': 0,
                    'algorithm_usage': {'Dijkstra': 0},
                    'total_latency': 0,
                    'total_energy_saved': 0
                }
            if 'cross_shard_tx_stats' not in self.stats:
                self.stats['cross_shard_tx_stats'] = {
                    'total_attempts': 0,
                    'successful': 0,
                    'failed': 0,
                    'energy_saved': 0
                }
                
            # Cập nhật số lần thử
            self.stats['cross_shard_tx_stats']['total_attempts'] = self.stats['cross_shard_tx_stats'].get('total_attempts', 0) + 1
            
            # Xác định shard nguồn và đích
            actual_source_shard_id = source_shard_id
            actual_target_shard_id = target_shard_id
            
            # Nếu không cung cấp source_shard_id, sử dụng từ transaction
            if actual_source_shard_id is None:
                if hasattr(transaction, 'source_shard'):
                    actual_source_shard_id = transaction.source_shard
                else:
                    print("Lỗi: Không thể xác định shard nguồn")
                    self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
                    return False
                    
            # Nếu không cung cấp target_shard_id, sử dụng từ transaction
            if actual_target_shard_id is None:
                if hasattr(transaction, 'target_shard'):
                    actual_target_shard_id = transaction.target_shard
                else:
                    print("Lỗi: Không thể xác định shard đích")
                    self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
                    return False
                    
            print(f"Xử lý giao dịch chéo shard từ {actual_source_shard_id} đến {actual_target_shard_id}")
            
            # Nếu nguồn và đích giống nhau, thực hiện trực tiếp
            if actual_source_shard_id == actual_target_shard_id:
                print(f"Nguồn và đích giống nhau ({actual_source_shard_id}), thực hiện trực tiếp")
                result = self._execute_transaction(transaction, actual_source_shard_id, actual_target_shard_id)
                if result:
                    self.stats['cross_shard_tx_stats']['successful'] = self.stats['cross_shard_tx_stats'].get('successful', 0) + 1
                else:
                    self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
                return result
            
            # Tìm đường đi tối ưu
            try:
                optimal_path = self._optimize_path(actual_source_shard_id, actual_target_shard_id, transaction)
                
                # Cập nhật thống kê về giao dịch đã tối ưu hóa
                if optimal_path and len(optimal_path) > 2:  # Nếu đường đi có nhiều hơn 2 shard (không phải đường đi trực tiếp)
                    if 'path_selection_stats' not in self.stats:
                        self.stats['path_selection_stats'] = {}
                    if 'optimized_transactions' not in self.stats['path_selection_stats']:
                        self.stats['path_selection_stats']['optimized_transactions'] = 0
                    self.stats['path_selection_stats']['optimized_transactions'] += 1
                    print(f"Đã tối ưu hóa giao dịch với đường đi: {optimal_path}")
                
                if optimal_path and len(optimal_path) > 0:
                    print(f"Tìm thấy đường đi tối ưu: {optimal_path}")
                    
                    # Thực hiện giao dịch theo đường đi tối ưu
                    current_shard = actual_source_shard_id
                    success = True
                    
                    for next_shard in optimal_path[1:]:  # Bỏ qua shard đầu tiên vì đó là nguồn
                        # Chuyển đổi tensor thành số nguyên nếu cần
                        if isinstance(current_shard, torch.Tensor):
                            current_shard = current_shard.item()
                        if isinstance(next_shard, torch.Tensor):
                            next_shard = next_shard.item()
                            
                        print(f"Chuyển giao dịch từ shard {current_shard} đến shard {next_shard}")
                        if not self._execute_transaction(transaction, current_shard, next_shard):
                            print(f"Lỗi khi chuyển từ shard {current_shard} đến shard {next_shard}")
                            success = False
                            break
                        current_shard = next_shard
                        
                    if success:
                        print(f"Giao dịch chéo shard thành công theo đường đi: {optimal_path}")
                        self.stats['cross_shard_tx_stats']['successful'] = self.stats['cross_shard_tx_stats'].get('successful', 0) + 1
                        
                        # Tính toán năng lượng tiết kiệm được
                        try:
                            direct_energy = self._calculate_energy_cost(actual_source_shard_id, actual_target_shard_id)
                            path_energy = 0
                            for i in range(len(optimal_path)-1):
                                if i < len(optimal_path) and i+1 < len(optimal_path):  # Kiểm tra chỉ số hợp lệ
                                    path_energy += self._calculate_energy_cost(optimal_path[i], optimal_path[i+1])
                            energy_saved = max(0, direct_energy - path_energy)
                            
                            self.stats['cross_shard_tx_stats']['energy_saved'] = self.stats['cross_shard_tx_stats'].get('energy_saved', 0) + energy_saved
                            print(f"Tiết kiệm được {energy_saved} đơn vị năng lượng")
                            
                            # Cập nhật thống kê với đường đi tối ưu
                            self._update_statistics(transaction, optimal_path, direct_energy - path_energy, energy_saved)
                            
                            # In thông tin debug về thống kê
                            print("DEBUG STATS:", self.stats)
                        except Exception as e:
                            print(f"Lỗi khi tính toán chi phí năng lượng: {e}")
                            energy_saved = 0
                            
                        return True
                    else:
                        print("Giao dịch chéo shard thất bại, thử thực hiện trực tiếp")
                else:
                    print("Không tìm thấy đường đi tối ưu, thử thực hiện trực tiếp")
                    
                # Nếu không tìm được đường đi tối ưu hoặc thất bại, thử thực hiện trực tiếp
                result = self._execute_transaction(transaction, actual_source_shard_id, actual_target_shard_id)
                if result:
                    self.stats['cross_shard_tx_stats']['successful'] = self.stats['cross_shard_tx_stats'].get('successful', 0) + 1
                    print("Giao dịch trực tiếp thành công")
                    # Cập nhật thống kê với đường đi trực tiếp
                    direct_path = [actual_source_shard_id, actual_target_shard_id]
                    self._update_statistics(transaction, direct_path, 0, 0)
                else:
                    self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
                    print("Giao dịch trực tiếp thất bại")
                    
                return result
                
            except Exception as e:
                print(f"Lỗi khi tối ưu đường đi: {e}")
                # Thử thực hiện trực tiếp nếu tối ưu thất bại
                result = self._execute_transaction(transaction, actual_source_shard_id, actual_target_shard_id)
                if result:
                    self.stats['cross_shard_tx_stats']['successful'] = self.stats['cross_shard_tx_stats'].get('successful', 0) + 1
                    print("Giao dịch trực tiếp thành công sau lỗi tối ưu")
                    # Cập nhật thống kê với đường đi trực tiếp
                    direct_path = [actual_source_shard_id, actual_target_shard_id]
                    self._update_statistics(transaction, direct_path, 0, 0)
                else:
                    self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
                    print("Giao dịch trực tiếp thất bại sau lỗi tối ưu")
                return result
            
        except Exception as e:
            print(f"Lỗi khi xử lý giao dịch chéo shard: {e}")
            self.stats['cross_shard_tx_stats']['failed'] = self.stats['cross_shard_tx_stats'].get('failed', 0) + 1
            return False

    def _compute_transaction_state(self, transaction, source_shard_id, target_shard_id):
        """
        Tính toán vector trạng thái cho giao dịch, sử dụng cho DQN agent
        
        Args:
            transaction: Giao dịch cần tính toán trạng thái
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            
        Returns:
            numpy.ndarray: Vector trạng thái
        """
        try:
            # Kiểm tra tính hợp lệ của shard IDs
            if source_shard_id is None or target_shard_id is None:
                # Trả về vector zero nếu không hợp lệ
                return np.zeros(10, dtype=np.float32)
                
            # Lấy thông tin cơ bản từ giao dịch
            tx_size = getattr(transaction, 'size', 1.0)  # Kích thước giao dịch
            tx_complexity = getattr(transaction, 'complexity', 1.0)  # Độ phức tạp
            tx_priority = getattr(transaction, 'priority', 1.0)  # Ưu tiên
            
            # Lấy thông tin shard nguồn
            source_shard = self.network.shards.get(source_shard_id)
            source_congestion = getattr(source_shard, 'congestion', 0.0) if source_shard else 0.0
            source_tx_pool = getattr(source_shard, 'transaction_pool', {})
            source_tx_count = len(source_tx_pool) if source_tx_pool is not None else 0
            
            # Lấy thông tin shard đích
            target_shard = self.network.shards.get(target_shard_id)
            target_congestion = getattr(target_shard, 'congestion', 0.0) if target_shard else 0.0
            target_tx_pool = getattr(target_shard, 'transaction_pool', {})
            target_tx_count = len(target_tx_pool) if target_tx_pool is not None else 0
            
            # Tính toán độ trễ trực tiếp giữa nguồn và đích
            direct_latency = self._estimate_direct_latency(source_shard_id, target_shard_id)
            
            # Tính toán khoảng cách giữa nguồn và đích
            distance = abs(source_shard_id - target_shard_id)
            
            # Kiểm tra nếu có shard embeddings
            similarity = 0.0
            if hasattr(self, 'shard_embeddings') and self.shard_embeddings is not None:
                if source_shard_id < len(self.shard_embeddings) and target_shard_id < len(self.shard_embeddings):
                    # Tính độ tương đồng cosine
                    source_embed = self.shard_embeddings[source_shard_id]
                    target_embed = self.shard_embeddings[target_shard_id]
                    
                    # Sử dụng F.cosine_similarity nếu tensor có đủ chiều
                    try:
                        if len(source_embed.shape) == 1 and len(target_embed.shape) == 1:
                            source_embed = source_embed.unsqueeze(0)
                            target_embed = target_embed.unsqueeze(0)
                        similarity = F.cosine_similarity(source_embed, target_embed).item()
                    except Exception as e:
                        print(f"Lỗi khi tính cosine similarity: {str(e)}")
                        # Tính theo công thức thủ công
                        dot_product = torch.dot(source_embed, target_embed).item()
                        source_norm = torch.norm(source_embed).item()
                        target_norm = torch.norm(target_embed).item()
                        if source_norm > 0 and target_norm > 0:
                            similarity = dot_product / (source_norm * target_norm)
                        else:
                            similarity = 0.0
            
            # Tạo vector trạng thái
            state = np.array([
                tx_size / 10.0,  # Chuẩn hóa kích thước giao dịch
                tx_complexity / 5.0,  # Chuẩn hóa độ phức tạp
                tx_priority / 3.0,  # Chuẩn hóa ưu tiên
                source_congestion,  # Tắc nghẽn shard nguồn
                target_congestion,  # Tắc nghẽn shard đích
                source_tx_count / 100.0,  # Chuẩn hóa số giao dịch nguồn
                target_tx_count / 100.0,  # Chuẩn hóa số giao dịch đích
                direct_latency / 100.0,  # Chuẩn hóa độ trễ
                distance / 10.0,  # Chuẩn hóa khoảng cách
                (similarity + 1) / 2.0  # Chuyển [-1, 1] thành [0, 1]
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            print(f"Lỗi trong _compute_transaction_state: {str(e)}")
            # Trả về vector zero nếu có lỗi
            return np.zeros(10, dtype=np.float32)

    def _optimize_path(self, source_shard_id, target_shard_id, transaction=None):
        """
        Tìm đường đi tối ưu giữa hai shard dựa trên độ trễ, tắc nghẽn và các yếu tố khác
        
        Args:
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            transaction: Giao dịch cần tối ưu đường đi (tùy chọn)
            
        Returns:
            list: Danh sách các shard IDs tạo thành đường đi tối ưu, hoặc None nếu không tìm thấy
        """
        try:
            # Khởi tạo thống kê nếu chưa có
            if not hasattr(self, 'stats'):
                self.stats = {}
            if 'path_selection_stats' not in self.stats:
                self.stats['path_selection_stats'] = {
                    'total_attempts': 0,
                    'successful_optimizations': 0,
                    'direct_paths': 0,
                    'algorithm_usage': {'Dijkstra': 0},
                    'total_latency': 0,
                    'total_energy_saved': 0,
                    'optimized_transactions': 0
                }
                
            # Cập nhật số lần thử
            self.stats['path_selection_stats']['total_attempts'] = self.stats['path_selection_stats'].get('total_attempts', 0) + 1
            
            # Kiểm tra tính hợp lệ của đầu vào
            if source_shard_id is None or target_shard_id is None:
                print(f"Lỗi: source_shard_id hoặc target_shard_id là None")
                return None
                
            # Nếu nguồn và đích giống nhau, trả về đường đi trực tiếp
            if source_shard_id == target_shard_id:
                self.stats['path_selection_stats']['direct_paths'] = self.stats['path_selection_stats'].get('direct_paths', 0) + 1
                return [source_shard_id]
                
            # Tạo ma trận độ trễ và tensor tắc nghẽn nếu chưa được khởi tạo
            if not hasattr(self, 'latency_matrix') or self.latency_matrix is None:
                print("Khởi tạo ma trận độ trễ")
                num_shards = len(self.network.shards)
                self.latency_matrix = torch.rand((num_shards, num_shards)) * 100  # Độ trễ ngẫu nhiên từ 0-100ms
                # Đặt độ trễ từ một shard đến chính nó là 0
                for i in range(num_shards):
                    self.latency_matrix[i, i] = 0
                    
            if not hasattr(self, 'congestion_tensor') or self.congestion_tensor is None:
                print("Khởi tạo tensor tắc nghẽn")
                num_shards = len(self.network.shards)
                self.congestion_tensor = torch.zeros((num_shards, num_shards))
                
            # Lấy số lượng shard
            num_shards = self.latency_matrix.shape[0]
            
            # Kiểm tra tính hợp lệ của chỉ số shard
            if source_shard_id >= num_shards or target_shard_id >= num_shards:
                print(f"Lỗi: Chỉ số shard không hợp lệ: {source_shard_id}, {target_shard_id}, num_shards={num_shards}")
                return [source_shard_id, target_shard_id]
            
            # Tạo ma trận trọng số cho thuật toán Dijkstra
            weight_matrix = torch.zeros((num_shards, num_shards))
            
            # Tính toán trọng số dựa trên độ trễ và tắc nghẽn
            for i in range(num_shards):
                for j in range(num_shards):
                    try:
                        if i == j:
                            weight_matrix[i, j] = 0  # Không có trọng số từ một shard đến chính nó
                        else:
                            # Trọng số là tổng hợp của độ trễ và tắc nghẽn
                            latency_weight = self.latency_matrix[i, j].item()
                            
                            # Đảm bảo congestion_tensor có đúng kích thước
                            congestion_weight = 0
                            if i < self.congestion_tensor.shape[0] and j < self.congestion_tensor.shape[1]:
                                congestion_weight = self.congestion_tensor[i, j].item() * 50  # Nhân với hệ số để tăng ảnh hưởng
                            
                            # Nếu có giao dịch, xem xét kích thước và độ phức tạp
                            transaction_weight = 0
                            if transaction is not None:
                                if hasattr(transaction, 'size'):
                                    transaction_weight += transaction.size * 0.1
                                if hasattr(transaction, 'complexity'):
                                    transaction_weight += transaction.complexity * 0.2
                                if hasattr(transaction, 'priority'):
                                    # Ưu tiên cao sẽ giảm trọng số
                                    transaction_weight -= transaction.priority * 0.3
                                    
                            # Tổng hợp trọng số
                            weight_matrix[i, j] = latency_weight + congestion_weight + transaction_weight
                    except Exception as e:
                        print(f"Lỗi khi tính toán trọng số cho ({i}, {j}): {e}")
                        # Đặt giá trị mặc định cao để tránh chọn đường đi này
                        weight_matrix[i, j] = 1000.0
            
            # Đảm bảo không có trọng số âm
            weight_matrix = torch.clamp(weight_matrix, min=0.1)
            
            # Thuật toán Dijkstra để tìm đường đi ngắn nhất
            distances = torch.full((num_shards,), float('inf'))
            distances[source_shard_id] = 0
            previous = torch.full((num_shards,), -1, dtype=torch.long)
            visited = torch.zeros(num_shards, dtype=torch.bool)
            
            for _ in range(num_shards):
                # Tìm shard chưa thăm có khoảng cách nhỏ nhất
                min_distance = float('inf')
                min_index = -1
                for i in range(num_shards):
                    if not visited[i] and distances[i] < min_distance:
                        min_distance = distances[i]
                        min_index = i
                        
                if min_index == -1 or min_index == target_shard_id:
                    break
                    
                visited[min_index] = True
                
                # Cập nhật khoảng cách cho các shard kề
                for i in range(num_shards):
                    if not visited[i] and weight_matrix[min_index, i] > 0:
                        new_distance = distances[min_index] + weight_matrix[min_index, i]
                        if new_distance < distances[i]:
                            distances[i] = new_distance
                            previous[i] = min_index
                            
            # Xây dựng đường đi từ nguồn đến đích
            if distances[target_shard_id] == float('inf'):
                print(f"Không tìm thấy đường đi từ shard {source_shard_id} đến shard {target_shard_id}")
                self.stats['path_selection_stats']['direct_paths'] = self.stats['path_selection_stats'].get('direct_paths', 0) + 1
                return [source_shard_id, target_shard_id]  # Trả về đường đi trực tiếp nếu không tìm thấy đường đi tối ưu
                
            # Xây dựng đường đi
            path = []
            current = target_shard_id
            while current != -1:
                path.append(current)
                current = previous[current]
                
            # Đảo ngược đường dẫn
            path = path[::-1]
            
            # Đảm bảo path chỉ chứa số nguyên, không phải tensor
            path = [p.item() if isinstance(p, torch.Tensor) else p for p in path]
            
            # Kiểm tra xem đường đi có bắt đầu từ nguồn không
            if path[0] != source_shard_id:
                print(f"Lỗi: Đường đi không bắt đầu từ shard nguồn {source_shard_id}, mà từ {path[0]}")
                path = [source_shard_id] + path
                
            # Cập nhật thống kê
            self.stats['path_selection_stats']['successful_optimizations'] = self.stats['path_selection_stats'].get('successful_optimizations', 0) + 1
            self.stats['path_selection_stats']['algorithm_usage']['Dijkstra'] = self.stats['path_selection_stats']['algorithm_usage'].get('Dijkstra', 0) + 1
            
            # Tính toán độ trễ và năng lượng tiết kiệm được
            total_latency = 0
            for i in range(len(path)-1):
                if i < len(path) and i+1 < len(path) and path[i] < self.latency_matrix.shape[0] and path[i+1] < self.latency_matrix.shape[1]:
                    total_latency += self.latency_matrix[path[i], path[i+1]].item()
                    
            # Đảm bảo chỉ số hợp lệ khi truy cập latency_matrix
            if source_shard_id < self.latency_matrix.shape[0] and target_shard_id < self.latency_matrix.shape[1]:
                direct_latency = self.latency_matrix[source_shard_id, target_shard_id].item()
            else:
                direct_latency = 100.0  # Giá trị mặc định nếu chỉ số không hợp lệ
                
            energy_saved = max(0, direct_latency - total_latency)
            
            self.stats['path_selection_stats']['total_latency'] = self.stats['path_selection_stats'].get('total_latency', 0) + total_latency
            self.stats['path_selection_stats']['total_energy_saved'] = self.stats['path_selection_stats'].get('total_energy_saved', 0) + energy_saved
            
            print(f"Tìm thấy đường đi tối ưu: {path}")
            return path
            
        except Exception as e:
            print(f"Lỗi khi tối ưu đường đi: {e}")
            # Trả về đường đi trực tiếp trong trường hợp lỗi
            self.stats['path_selection_stats']['direct_paths'] = self.stats['path_selection_stats'].get('direct_paths', 0) + 1
            return [source_shard_id, target_shard_id]

    def _estimate_direct_latency(self, source_shard_id, target_shard_id):
        """
        Ước tính độ trễ của đường dẫn trực tiếp giữa hai shard
        
        Args:
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            
        Returns:
            float: Độ trễ ước tính (ms)
        """
        # Kiểm tra tính hợp lệ của IDs
        if source_shard_id is None or target_shard_id is None:
            return 100.0  # Giá trị mặc định cao
        
        # Kiểm tra nếu có latency matrix
        if hasattr(self, 'latency_matrix') and self.latency_matrix is not None:
            # Đảm bảo index trong khoảng hợp lệ
            if (0 <= source_shard_id < self.latency_matrix.shape[0] and 
                0 <= target_shard_id < self.latency_matrix.shape[1]):
                # Lấy độ trễ từ ma trận và chuyển về ms
                latency = self.latency_matrix[source_shard_id, target_shard_id].item() * 50.0
                return latency
        
        # Nếu không có ma trận, tính dựa trên khoảng cách
        distance = abs(source_shard_id - target_shard_id)
        base_latency = 50.0  # Độ trễ cơ bản (ms)
        distance_factor = distance * 10.0  # 10ms cho mỗi đơn vị khoảng cách
        
        # Thêm yếu tố tắc nghẽn nếu có
        congestion_factor = 0.0
        if hasattr(self, 'congestion_tensor') and self.congestion_tensor is not None:
            if (0 <= source_shard_id < len(self.congestion_tensor) and 
                0 <= target_shard_id < len(self.congestion_tensor)):
                source_congestion = self.congestion_tensor[source_shard_id].item()
                target_congestion = self.congestion_tensor[target_shard_id].item()
                congestion_factor = (source_congestion + target_congestion) * 20.0  # 0-40ms dựa trên tắc nghẽn
        
        # Tổng độ trễ
        total_latency = base_latency + distance_factor + congestion_factor
        return total_latency
        
    def _estimate_path_latency(self, path):
        """
        Ước tính độ trễ của một đường dẫn đa bước
        
        Args:
            path: Danh sách các shard IDs trong đường dẫn
            
        Returns:
            float: Tổng độ trễ ước tính (ms)
        """
        if not path or len(path) < 2:
            return 100.0  # Giá trị mặc định cao cho đường dẫn không hợp lệ
        
        total_latency = 0.0
        
        # Tính tổng độ trễ cho từng segment trong đường dẫn
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i+1]
            segment_latency = self._estimate_direct_latency(source, target)
            total_latency += segment_latency
            
            # Thêm độ trễ xử lý tại các nút trung gian
            if 0 < i < len(path) - 1:  # Không tính nút nguồn và đích
                processing_latency = 5.0  # Độ trễ xử lý cơ bản (ms)
                
                # Thêm phần phụ thuộc vào tắc nghẽn
                if hasattr(self, 'congestion_tensor') and self.congestion_tensor is not None:
                    if 0 <= source < len(self.congestion_tensor):
                        congestion = self.congestion_tensor[source].item()
                        processing_latency += congestion * 15.0  # 0-15ms thêm dựa trên tắc nghẽn
                        
                total_latency += processing_latency
                
        return total_latency 

    def _update_statistics(self, transaction, path, latency, energy_saved):
        """Cập nhật thống kê cho giao dịch"""
        # Khởi tạo thống kê nếu chưa tồn tại
        if not hasattr(self, 'stats'):
            self.stats = {}
            
        if "path_selection_stats" not in self.stats:
            self.stats["path_selection_stats"] = {
                "total_attempts": 0,
                "successful_optimizations": 0,
                "direct_paths": 0,
                "algorithm_usage": {"Dijkstra": 0},
                "total_latency": 0,
                "total_energy_saved": 0,
                "cross_shard_transactions": 0,
                "optimized_transactions": 0
            }
        
        # Chuyển đổi các phần tử trong path từ tensor sang số nguyên nếu cần
        path = [p.item() if isinstance(p, torch.Tensor) else p for p in path]
        
        # Cập nhật thống kê về đường đi
        if len(path) == 2:
            self.stats["path_selection_stats"]["direct_paths"] = self.stats["path_selection_stats"].get("direct_paths", 0) + 1
        else:
            self.stats["path_selection_stats"]["successful_optimizations"] = self.stats["path_selection_stats"].get("successful_optimizations", 0) + 1
            
        # Cập nhật thống kê về giao dịch xuyên shard
        source_shard_id = getattr(transaction, 'source_shard', path[0])
        target_shard_id = getattr(transaction, 'target_shard', path[-1])
        
        # Chuyển đổi source_shard_id và target_shard_id từ tensor sang số nguyên nếu cần
        if isinstance(source_shard_id, torch.Tensor):
            source_shard_id = source_shard_id.item()
        if isinstance(target_shard_id, torch.Tensor):
            target_shard_id = target_shard_id.item()
        
        if source_shard_id != target_shard_id:
            self.stats["path_selection_stats"]["cross_shard_transactions"] = self.stats["path_selection_stats"].get("cross_shard_transactions", 0) + 1
            if len(path) > 2:
                self.stats["path_selection_stats"]["optimized_transactions"] = self.stats["path_selection_stats"].get("optimized_transactions", 0) + 1
                print(f"Đã tối ưu hóa giao dịch với đường đi: {path}")
            
        # Cập nhật thống kê về độ trễ và năng lượng
        self.stats["path_selection_stats"]["total_latency"] = self.stats["path_selection_stats"].get("total_latency", 0) + latency
        self.stats["path_selection_stats"]["total_energy_saved"] = self.stats["path_selection_stats"].get("total_energy_saved", 0) + energy_saved
        
        # Cập nhật thống kê về thuật toán sử dụng
        algorithm = getattr(transaction, 'algorithm_used', 'Dijkstra')
        if "algorithm_usage" not in self.stats["path_selection_stats"]:
            self.stats["path_selection_stats"]["algorithm_usage"] = {}
        if algorithm not in self.stats["path_selection_stats"]["algorithm_usage"]:
            self.stats["path_selection_stats"]["algorithm_usage"][algorithm] = 0
        self.stats["path_selection_stats"]["algorithm_usage"][algorithm] = self.stats["path_selection_stats"]["algorithm_usage"].get(algorithm, 0) + 1
            
        # Cập nhật thống kê tổng thể
        if "total_transactions" not in self.stats:
            self.stats["total_transactions"] = 0
        if "successful_transactions" not in self.stats:
            self.stats["successful_transactions"] = 0
        if "total_cross_shard" not in self.stats:
            self.stats["total_cross_shard"] = 0
        if "successful_cross_shard" not in self.stats:
            self.stats["successful_cross_shard"] = 0
            
        self.stats["total_transactions"] = self.stats.get("total_transactions", 0) + 1
        self.stats["successful_transactions"] = self.stats.get("successful_transactions", 0) + 1
        
        if source_shard_id != target_shard_id:
            self.stats["total_cross_shard"] = self.stats.get("total_cross_shard", 0) + 1
            self.stats["successful_cross_shard"] = self.stats.get("successful_cross_shard", 0) + 1
            
        # Cập nhật thống kê về hiệu suất
        if "performance_metrics" not in self.stats:
            self.stats["performance_metrics"] = {
                "total_latency": 0,
                "total_energy": 0,
                "total_optimizations": 0
            }
            
        self.stats["performance_metrics"]["total_latency"] = self.stats["performance_metrics"].get("total_latency", 0) + latency
        self.stats["performance_metrics"]["total_energy"] = self.stats["performance_metrics"].get("total_energy", 0) + energy_saved
        if len(path) > 2:
            self.stats["performance_metrics"]["total_optimizations"] = self.stats["performance_metrics"].get("total_optimizations", 0) + 1

    def _execute_transaction(self, transaction, source_shard_id, target_shard_id):
        """
        Thực hiện giao dịch giữa hai shard
        
        Args:
            transaction: Giao dịch cần thực hiện
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            # Kiểm tra tính hợp lệ của shard
            if source_shard_id is None or target_shard_id is None:
                print(f"Lỗi: source_shard_id hoặc target_shard_id là None")
                return False
                
            # Lấy thông tin shard
            source_shard = self.network.shards.get(source_shard_id)
            target_shard = self.network.shards.get(target_shard_id)
            
            if not source_shard or not target_shard:
                print(f"Lỗi: Không tìm thấy shard {source_shard_id} hoặc {target_shard_id}")
                return False
            
            # Gán nguồn và đích cho giao dịch nếu chưa có
            if not hasattr(transaction, 'source_shard') or transaction.source_shard is None:
                transaction.source_shard = source_shard_id
            if not hasattr(transaction, 'target_shard') or transaction.target_shard is None:
                transaction.target_shard = target_shard_id
                
            # Kiểm tra số dư
            if hasattr(transaction, 'amount') and source_shard_id == transaction.source_shard:
                if hasattr(source_shard, 'check_balance') and callable(getattr(source_shard, 'check_balance', None)):
                    if not source_shard.check_balance(transaction.amount):
                        print(f"Lỗi: Shard {source_shard_id} không đủ số dư")
                        return False
                    
            # Thực hiện giao dịch
            if hasattr(transaction, 'amount'):
                if source_shard_id == transaction.source_shard and hasattr(source_shard, 'update_balance') and callable(getattr(source_shard, 'update_balance', None)):
                    source_shard.update_balance(-transaction.amount)
                    print(f"Đã trừ {transaction.amount} từ shard {source_shard_id}")
                    
                if target_shard_id == transaction.target_shard and hasattr(target_shard, 'update_balance') and callable(getattr(target_shard, 'update_balance', None)):
                    target_shard.update_balance(transaction.amount)
                    print(f"Đã thêm {transaction.amount} vào shard {target_shard_id}")
                
            # Cập nhật trạng thái giao dịch
            try:
                if hasattr(transaction, 'update_status') and callable(transaction.update_status):
                    from dqn_blockchain_sim.blockchain.transaction import TransactionStatus
                    transaction.update_status(TransactionStatus.CONFIRMED)
                    print(f"Đã cập nhật trạng thái giao dịch thành CONFIRMED")
            except Exception as e:
                print(f"Cảnh báo: Không thể cập nhật trạng thái giao dịch: {e}")
            
            # Cập nhật thống kê cho shard
            if not hasattr(source_shard, 'successful_tx_count'):
                source_shard.successful_tx_count = 0
            source_shard.successful_tx_count += 1
            
            if not hasattr(target_shard, 'successful_tx_count'):
                target_shard.successful_tx_count = 0
            target_shard.successful_tx_count += 1
            
            print(f"Giao dịch từ shard {source_shard_id} đến shard {target_shard_id} thành công!")
            return True
            
        except Exception as e:
            print(f"Lỗi khi thực hiện giao dịch: {e}")
            return False

    def connect_to_network(self, network):
        """
        Kết nối MAD-RAPID với mạng blockchain
        
        Args:
            network: Đối tượng mạng blockchain
        """
        print("Đang kết nối MAD-RAPID với mạng blockchain...")
        self.network = network
        
        # Khởi tạo thống kê nếu chưa có
        if not hasattr(self, 'stats'):
            self.stats = {}
        if 'path_selection_stats' not in self.stats:
            self.stats['path_selection_stats'] = {
                'total_attempts': 0,
                'successful_optimizations': 0,
                'direct_paths': 0,
                'algorithm_usage': {'Dijkstra': 0},
                'total_latency': 0,
                'total_energy_saved': 0
            }
        if 'cross_shard_tx_stats' not in self.stats:
            self.stats['cross_shard_tx_stats'] = {
                'total_attempts': 0,
                'successful': 0,
                'failed': 0,
                'energy_saved': 0
            }
            
        # Lấy số lượng shard
        num_shards = len(self.network.shards)
        print(f"Số shard trong MAD-RAPID: {num_shards}")
        
        # Khởi tạo ma trận độ trễ
        self.latency_matrix = torch.rand((num_shards, num_shards)) * 100  # Độ trễ ngẫu nhiên từ 0-100ms
        # Đặt độ trễ từ một shard đến chính nó là 0
        for i in range(num_shards):
            self.latency_matrix[i, i] = 0
            
        # Khởi tạo tensor tắc nghẽn
        self.congestion_tensor = torch.zeros((num_shards, num_shards))
        
        # Cập nhật tensor tắc nghẽn ban đầu
        for i in range(num_shards):
            congestion = self._predict_congestion(i)
            for j in range(num_shards):
                if i != j:
                    self.congestion_tensor[i, j] = congestion
        
        # Khởi tạo shard embeddings
        print("Đang khởi tạo shard_embeddings...")
        self.shard_embeddings = torch.rand((num_shards, 10))  # Mỗi shard có một embedding 10 chiều
        
        print("Đã kết nối MAD-RAPID với mạng blockchain thành công!")

    def _calculate_energy_cost(self, source_shard_id, target_shard_id):
        """
        Tính toán chi phí năng lượng cho việc truyền giao dịch giữa hai shard
        
        Args:
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            
        Returns:
            float: Chi phí năng lượng ước tính
        """
        try:
            # Kiểm tra tính hợp lệ của IDs
            if source_shard_id is None or target_shard_id is None:
                return 100.0  # Giá trị mặc định cao
                
            # Nếu nguồn và đích giống nhau, chi phí thấp
            if source_shard_id == target_shard_id:
                return 10.0
                
            # Đảm bảo latency_matrix đã được khởi tạo
            if not hasattr(self, 'latency_matrix') or self.latency_matrix is None:
                print("Khởi tạo ma trận độ trễ trong _calculate_energy_cost")
                num_shards = len(self.network.shards)
                self.latency_matrix = torch.rand((num_shards, num_shards)) * 100
                for i in range(num_shards):
                    self.latency_matrix[i, i] = 0
                    
            # Đảm bảo congestion_tensor đã được khởi tạo
            if not hasattr(self, 'congestion_tensor') or self.congestion_tensor is None:
                print("Khởi tạo tensor tắc nghẽn trong _calculate_energy_cost")
                num_shards = len(self.network.shards)
                self.congestion_tensor = torch.zeros((num_shards, num_shards))
                
            # Kiểm tra tính hợp lệ của chỉ số
            num_shards = self.latency_matrix.shape[0]
            
            # Chuyển đổi tensor thành số nguyên nếu cần
            if isinstance(source_shard_id, torch.Tensor):
                source_shard_id = source_shard_id.item()
            if isinstance(target_shard_id, torch.Tensor):
                target_shard_id = target_shard_id.item()
                
            if source_shard_id >= num_shards or target_shard_id >= num_shards:
                print(f"Lỗi: Chỉ số shard không hợp lệ trong _calculate_energy_cost: {source_shard_id}, {target_shard_id}, num_shards={num_shards}")
                return 100.0
                
            # Tính toán chi phí dựa trên độ trễ và tắc nghẽn
            base_cost = 50.0  # Chi phí cơ bản
            
            # Thêm chi phí dựa trên độ trễ
            latency_cost = 0.0
            try:
                if source_shard_id < self.latency_matrix.shape[0] and target_shard_id < self.latency_matrix.shape[1]:
                    latency_cost = self.latency_matrix[source_shard_id, target_shard_id].item() * 0.5
            except Exception as e:
                print(f"Lỗi khi tính toán chi phí độ trễ: {e}")
                latency_cost = 50.0  # Giá trị mặc định nếu có lỗi
                
            # Thêm chi phí dựa trên tắc nghẽn
            congestion_cost = 0.0
            try:
                if source_shard_id < self.congestion_tensor.shape[0] and target_shard_id < self.congestion_tensor.shape[1]:
                    congestion_cost = self.congestion_tensor[source_shard_id, target_shard_id].item() * 100.0
            except Exception as e:
                print(f"Lỗi khi tính toán chi phí tắc nghẽn: {e}")
                
            # Tổng chi phí
            total_cost = base_cost + latency_cost + congestion_cost
            
            # Thêm yếu tố ngẫu nhiên nhỏ để tránh đồng nhất
            random_factor = random.uniform(0.9, 1.1)
            total_cost *= random_factor
            
            return total_cost
            
        except Exception as e:
            print(f"Lỗi khi tính toán chi phí năng lượng: {e}")
            return 100.0  # Giá trị mặc định trong trường hợp lỗi