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
from collections import defaultdict
from dqn_blockchain_sim.blockchain.network import BlockchainNetwork

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
        Forward pass của mô hình
        
        Args:
            x: Tensor đầu vào batch_size x sequence_length x input_size
            hidden: Trạng thái ẩn ban đầu (h0, c0)
            
        Returns:
            outputs: Dự đoán tắc nghẽn
            hidden: Trạng thái ẩn cuối cùng
        """
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Lấy kết quả của bước thời gian cuối cùng
        last_time_step = lstm_out[:, -1, :]
        
        # Áp dụng lớp fully-connected để dự đoán
        output = self.fc(last_time_step)
        output = torch.sigmoid(output)  # Chuyển đổi thành phạm vi [0, 1]
        
        return output, hidden
    
    def predict(self, features: np.ndarray) -> float:
        """
        Dự đoán mức độ tắc nghẽn dựa trên đặc trưng
        
        Args:
            features: Mảng đặc trưng của shard
            
        Returns:
            Mức độ tắc nghẽn dự đoán
        """
        self.eval()  # Đặt mô hình ở chế độ đánh giá
        
        with torch.no_grad():
            # Chuyển đổi đặc trưng thành tensor
            x = torch.FloatTensor(features).unsqueeze(0)  # Thêm batch dimension
            
            # Dự đoán
            prediction, _ = self.forward(x)
            
            # Trả về kết quả
            return prediction.item()


class AttentionBasedPathOptimizer:
    """
    Tối ưu hóa đường dẫn xuyên mảnh dựa trên cơ chế Attention
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Khởi tạo bộ tối ưu đường dẫn dựa trên Attention
        
        Args:
            embedding_dim: Kích thước vector biểu diễn của mỗi shard
        """
        self.embedding_dim = embedding_dim
        
        # Khởi tạo các tham số cho cơ chế Attention
        self.query_transform = nn.Linear(embedding_dim, embedding_dim)
        self.key_transform = nn.Linear(embedding_dim, embedding_dim)
        self.value_transform = nn.Linear(embedding_dim, embedding_dim)
        
        # Đầu ra cuối cùng
        self.output_transform = nn.Linear(embedding_dim, 1)
        
    def compute_attention_weights(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Tính toán trọng số chú ý (attention weights)
        
        Args:
            query: Tensor truy vấn (source shard) batch_size x embedding_dim
            keys: Tensor khóa (các shard tiềm năng) batch_size x num_shards x embedding_dim
            
        Returns:
            Tensor trọng số chú ý batch_size x num_shards
        """
        # Chuyển đổi query và keys
        q = self.query_transform(query).unsqueeze(1)  # batch_size x 1 x embedding_dim
        k = self.key_transform(keys)  # batch_size x num_shards x embedding_dim
        
        # Tính điểm chú ý
        attention_scores = torch.bmm(q, k.transpose(1, 2))  # batch_size x 1 x num_shards
        
        # Chuẩn hóa điểm bằng softmax
        attention_weights = F.softmax(attention_scores / np.sqrt(self.embedding_dim), dim=2)
        
        return attention_weights.squeeze(1)  # batch_size x num_shards
    
    def find_optimal_path(self, 
                      source_embedding: torch.Tensor, 
                      shard_embeddings: torch.Tensor, 
                      congestion_levels: torch.Tensor,
                      latency_matrix: torch.Tensor) -> List[int]:
        """
        Tìm đường dẫn tối ưu giữa các shard
        
        Args:
            source_embedding: Vector biểu diễn shard nguồn
            shard_embeddings: Tensor biểu diễn các shard khác
            congestion_levels: Mức độ tắc nghẽn của các shard
            latency_matrix: Ma trận độ trễ giữa các shard
            
        Returns:
            Danh sách shard IDs tạo thành đường dẫn tối ưu
        """
        with torch.no_grad():
            # Tính toán trọng số chú ý
            attention_weights = self.compute_attention_weights(
                source_embedding.unsqueeze(0),  # Thêm batch dimension
                shard_embeddings.unsqueeze(0)   # Thêm batch dimension
            )
            
            # Kết hợp trọng số chú ý với mức độ tắc nghẽn
            combined_scores = attention_weights * (1 - congestion_levels.unsqueeze(0))
            
            # Tìm đường dẫn dùng thuật toán Dijkstra có điều chỉnh
            # (Đây là phiên bản đơn giản hóa)
            num_shards = shard_embeddings.size(0)
            path = self._dijkstra_search(
                combined_scores.squeeze(0).cpu().numpy(),
                latency_matrix.cpu().numpy(),
                0,  # Shard nguồn (giả định là 0)
                num_shards - 1  # Shard đích (giả định là cuối cùng)
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
            attention_scores: Điểm chú ý cho mỗi shard
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
                    
                    # Độ trễ được điều chỉnh bởi điểm chú ý (attention score)
                    adj_latency = latency_matrix[min_node, neighbor] / max(0.1, attention_scores[neighbor])
                    
                    if distance[min_node] + adj_latency < distance[neighbor]:
                        distance[neighbor] = distance[min_node] + adj_latency
                        parent[neighbor] = min_node
        
        # Xây dựng đường dẫn từ nguồn đến đích
        path = []
        current = target
        
        while current != -1:
            path.append(current)
            current = parent[current]
            
        # Đảo ngược đường dẫn
        return path[::-1]


class MADRAPIDProtocol:
    """
    Giao thức MAD-RAPID: Multi-Agent Dynamic - Routing with Adaptive
    Prioritization in Distributed Shards
    
    Giao thức này triển khai định tuyến thông minh dựa trên dự đoán tắc nghẽn
    và tối ưu hóa đường dẫn.
    """
    
    def __init__(self, num_shards=8, network=None, prediction_horizon: int = 5, embedding_dim: int = 64):
        """
        Khởi tạo giao thức MAD-RAPID
        
        Args:
            num_shards: Số lượng shard trong mạng
            network: Đối tượng mạng blockchain
            prediction_horizon: Số bước dự đoán tương lai
            embedding_dim: Kích thước vector embedding cho các shard
        """
        self.network = network
        self.num_shards = num_shards
        self.prediction_horizon = prediction_horizon
        self.embedding_dim = embedding_dim
        
        # Khởi tạo trạng thái của các shard
        self.shard_states = {i: {'congestion': 0.0, 'latency': 10.0} for i in range(num_shards)}
        
        # Khởi tạo bộ dự đoán tắc nghẽn (một cho mỗi shard)
        self.congestion_predictors = {
            i: LSTMCongestionPredictor(input_size=8, hidden_size=128, num_layers=2)
            for i in range(num_shards)
        }
        
        # Khởi tạo bộ tối ưu hóa đường dẫn
        self.path_optimizer = AttentionBasedPathOptimizer(embedding_dim=embedding_dim)
        
        # Lưu trữ lịch sử đặc trưng của mỗi shard để dự đoán
        self.feature_history = {i: [] for i in range(num_shards)}
        
        # Khởi tạo embedding của các shard
        self.shard_embeddings = torch.randn(num_shards, embedding_dim)
        
        # Ma trận độ trễ giữa các shard (ms)
        self.latency_matrix = torch.ones(num_shards, num_shards) * 50.0
        for i in range(num_shards):
            self.latency_matrix[i, i] = 0.0
            
        # Thông tin về đường dẫn đã tối ưu
        self.optimized_paths = {}
        
        # Thống kê hiệu suất
        self.performance_stats = {
            'total_transactions': 0,
            'optimized_transactions': 0,
            'latency_improvement': 0.0,
            'energy_saved': 0.0
        }
        
        if network:
            self._initialize_shard_embeddings()
        
    def _initialize_shard_embeddings(self) -> None:
        """
        Khởi tạo vector biểu diễn ban đầu cho các shard
        """
        num_shards = len(self.network.shards)
        
        # Tạo ma trận kết nối shard
        self.latency_matrix = np.zeros((num_shards, num_shards))
        
        # Lấy thông tin độ trễ từ shard_graph
        for i, j, data in self.network.shard_graph.edges(data=True):
            self.latency_matrix[i, j] = data.get('latency', 100)  # Giá trị mặc định 100ms
            self.latency_matrix[j, i] = data.get('latency', 100)  # Kết nối hai chiều
            
        # Khởi tạo vector biểu diễn shard
        for shard_id, shard in self.network.shards.items():
            # Vector biểu diễn ban đầu dựa trên thuộc tính shard
            initial_features = np.array([
                shard_id / num_shards,  # ID được chuẩn hóa
                len(shard.nodes) / 100,  # Số lượng nút (chuẩn hóa)
                shard.congestion_level,  # Mức độ tắc nghẽn
                shard.avg_latency / 1000 if hasattr(shard, 'avg_latency') else 0.1,  # Độ trễ trung bình
                len(shard.transaction_pool) / 1000,  # Kích thước pool
                len(self.network.shard_graph.edges(shard_id)) / num_shards,  # Số lượng kết nối
                random.random() * 0.1  # Nhiễu ngẫu nhiên để tạo sự khác biệt
            ])
            
            # Mở rộng vector lên kích thước embedding_dim
            padded_features = np.pad(
                initial_features,
                (0, self.embedding_dim - len(initial_features)),
                'constant'
            )
            
            # Lưu trữ vector biểu diễn
            self.shard_embeddings[shard_id] = torch.FloatTensor(padded_features)
            
    def update_feature_history(self) -> None:
        """
        Cập nhật lịch sử đặc trưng cho mỗi shard
        """
        current_time = time.time()
        
        for shard_id, shard in self.network.shards.items():
            # Thu thập đặc trưng của shard
            features = np.array([
                shard.congestion_level,
                len(shard.transaction_pool) / 1000,
                len(shard.cross_shard_queue) / 100,
                shard.avg_latency / 1000 if hasattr(shard, 'avg_latency') else 0.1,
                shard.total_transactions / 10000 if hasattr(shard, 'total_transactions') else 0.1,
                shard.confirmed_transactions / 10000 if hasattr(shard, 'confirmed_transactions') else 0.1,
                shard.energy_consumption / 10 if hasattr(shard, 'energy_consumption') else 0.1,
                current_time % 86400 / 86400  # Thời gian trong ngày (chuẩn hóa)
            ])
            
            # Thêm vào lịch sử
            self.feature_history[shard_id].append(features)
            
            # Giới hạn kích thước lịch sử
            if len(self.feature_history[shard_id]) > self.max_history_len:
                self.feature_history[shard_id] = self.feature_history[shard_id][-self.max_history_len:]
                
    def predict_congestion(self, shard_id: int) -> float:
        """
        Dự đoán mức độ tắc nghẽn trong tương lai cho một shard
        
        Args:
            shard_id: ID của shard
            
        Returns:
            Mức độ tắc nghẽn dự đoán
        """
        # Kiểm tra xem có đủ dữ liệu lịch sử không
        if shard_id not in self.feature_history or len(self.feature_history[shard_id]) < 3:
            # Không đủ dữ liệu, trả về mức độ tắc nghẽn hiện tại
            return self.network.shards[shard_id].congestion_level
            
        # Chuẩn bị dữ liệu đầu vào cho mô hình dự đoán
        features = np.array(self.feature_history[shard_id])
        
        # Dự đoán bằng mô hình
        predicted_congestion = self.congestion_predictors[shard_id].predict(features)
        
        return predicted_congestion
    
    def predict_all_shard_congestion(self) -> Dict[int, float]:
        """
        Dự đoán mức độ tắc nghẽn cho tất cả các shard
        
        Returns:
            Từ điển ánh xạ từ shard_id đến mức độ tắc nghẽn dự đoán
        """
        # Cập nhật lịch sử đặc trưng
        self.update_feature_history()
        
        # Dự đoán mức độ tắc nghẽn cho từng shard
        predictions = {}
        for shard_id in self.network.shards.keys():
            predictions[shard_id] = self.predict_congestion(shard_id)
            
        return predictions
    
    def optimize_cross_shard_path(self, 
                               source_shard_id: int, 
                               target_shard_id: int, 
                               transaction_size: int = 2048) -> List[int]:
        """
        Tối ưu hóa đường đi xuyên mảnh dựa trên dự báo tắc nghẽn
        
        Args:
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            transaction_size: Kích thước giao dịch (byte)
            
        Returns:
            Danh sách các shard IDs tạo thành đường dẫn tối ưu
        """
        # Kiểm tra bộ nhớ đệm
        cache_key = f"{source_shard_id}_{target_shard_id}_{transaction_size}"
        current_time = time.time()
        
        if (cache_key in self.optimized_paths and 
            current_time - self.route_cache_timestamp.get(cache_key, 0) < self.route_cache_ttl):
            return self.optimized_paths[cache_key]
            
        # Dự đoán mức độ tắc nghẽn
        congestion_predictions = self.predict_all_shard_congestion()
        
        # Chuyển đổi dự đoán thành tensor
        congestion_tensor = torch.FloatTensor([
            congestion_predictions.get(i, 0.5) 
            for i in range(len(self.network.shards))
        ])
        
        # Tập hợp tất cả các vector biểu diễn shard thành một tensor
        all_embeddings = torch.stack([
            self.shard_embeddings[i] 
            for i in range(len(self.network.shards))
        ])
        
        # Chuyển ma trận độ trễ thành tensor
        latency_tensor = torch.FloatTensor(self.latency_matrix)
        
        # Tìm đường dẫn tối ưu
        path = self.path_optimizer.find_optimal_path(
            self.shard_embeddings[source_shard_id],
            all_embeddings,
            congestion_tensor,
            latency_tensor
        )
        
        # Lưu vào bộ nhớ đệm
        self.optimized_paths[cache_key] = path
        self.route_cache_timestamp[cache_key] = current_time
        
        return path
    
    def route_transaction(self, transaction, source_shard_id: int, target_shard_id: int) -> List[int]:
        """
        Định tuyến giao dịch giữa các shard
        
        Args:
            transaction: Giao dịch cần định tuyến
            source_shard_id: ID shard nguồn
            target_shard_id: ID shard đích
            
        Returns:
            Danh sách các shard IDs tạo thành đường dẫn
        """
        # Ước tính kích thước giao dịch
        tx_size = transaction.data.get('size', 2048) if hasattr(transaction, 'data') else 2048
        
        # Tìm đường dẫn tối ưu
        path = self.optimize_cross_shard_path(source_shard_id, target_shard_id, tx_size)
        
        return path
    
    def compress_transaction(self, transaction, congestion_level: float) -> Any:
        """
        Nén giao dịch dựa trên mức độ tắc nghẽn
        
        Args:
            transaction: Giao dịch cần nén
            congestion_level: Mức độ tắc nghẽn của đường dẫn
            
        Returns:
            Giao dịch đã được nén
        """
        # Mức độ nén dựa trên mức độ tắc nghẽn
        if congestion_level > 0.8:
            # Nén cao cho tắc nghẽn cao
            compression_level = 3  # Tương ứng với mức nén cao nhất
        elif congestion_level > 0.5:
            # Nén trung bình
            compression_level = 2
        else:
            # Nén nhẹ hoặc không nén
            compression_level = 1
            
        # Trong mô phỏng, chúng ta chỉ thiết lập cấp độ nén
        # Trong triển khai thực tế, thực hiện nén thực sự ở đây
        if hasattr(transaction, 'data'):
            if isinstance(transaction.data, dict):
                transaction.data['compression_level'] = compression_level
            else:
                transaction.data = {'compression_level': compression_level}
                
        return transaction
    
    def process_cross_shard_transaction(self, transaction) -> bool:
        """
        Xử lý giao dịch xuyên mảnh
        
        Args:
            transaction: Giao dịch xuyên mảnh cần xử lý
            
        Returns:
            True nếu xử lý thành công, False nếu có lỗi
        """
        if not hasattr(transaction, 'source_shard') or not hasattr(transaction, 'target_shard'):
            return False
            
        source_shard_id = transaction.source_shard
        target_shard_id = transaction.target_shard
        
        # Tìm đường dẫn tối ưu
        path = self.route_transaction(transaction, source_shard_id, target_shard_id)
        
        if not path or len(path) < 2:  # Cần ít nhất shard nguồn và shard đích
            return False
            
        # Tính mức độ tắc nghẽn trung bình trên đường dẫn
        congestion_sum = 0
        for shard_id in path:
            if shard_id in self.network.shards:
                congestion_sum += self.network.shards[shard_id].congestion_level
                
        avg_congestion = congestion_sum / len(path)
        
        # Nén giao dịch nếu cần
        compressed_transaction = self.compress_transaction(transaction, avg_congestion)
        
        # Gửi giao dịch qua đường dẫn tối ưu
        current_shard_id = source_shard_id
        
        for next_shard_id in path[1:]:  # Bỏ qua shard nguồn
            # Lấy shard hiện tại và shard tiếp theo
            current_shard = self.network.shards.get(current_shard_id)
            next_shard = self.network.shards.get(next_shard_id)
            
            if not current_shard or not next_shard:
                return False
                
            # Mô phỏng việc chuyển giao dịch giữa các shard
            if next_shard_id == target_shard_id:
                # Đã đến shard đích cuối cùng
                next_shard.add_transaction(compressed_transaction)
                return True
            else:
                # Chuyển qua shard trung gian
                # Trong triển khai thực tế, cần cập nhật trạng thái giao dịch
                # và theo dõi sự di chuyển của nó
                current_shard_id = next_shard_id
                
        return True 