import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import heapq

class MADRAPIDRouter:
    """
    Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution (MAD-RAPID).
    Thuật toán định tuyến thông minh cho giao dịch xuyên shard trong mạng blockchain.
    """
    
    def __init__(self, 
                 network: nx.Graph,
                 shards: List[List[int]],
                 congestion_weight: float = 0.4,
                 latency_weight: float = 0.3,
                 energy_weight: float = 0.2,
                 trust_weight: float = 0.1,
                 prediction_horizon: int = 3,
                 congestion_threshold: float = 0.7):
        """
        Khởi tạo bộ định tuyến MAD-RAPID.
        
        Args:
            network: Đồ thị mạng blockchain
            shards: Danh sách các shard và node trong mỗi shard
            congestion_weight: Trọng số cho mức độ tắc nghẽn
            latency_weight: Trọng số cho độ trễ
            energy_weight: Trọng số cho tiêu thụ năng lượng
            trust_weight: Trọng số cho điểm tin cậy
            prediction_horizon: Số bước dự đoán tắc nghẽn trong tương lai
            congestion_threshold: Ngưỡng tắc nghẽn để coi là tắc nghẽn
        """
        self.network = network
        self.shards = shards
        self.num_shards = len(shards)
        
        # Trọng số cho các yếu tố khác nhau trong quyết định định tuyến
        self.congestion_weight = congestion_weight
        self.latency_weight = latency_weight
        self.energy_weight = energy_weight
        self.trust_weight = trust_weight
        
        # Tham số cho dự đoán tắc nghẽn
        self.prediction_horizon = prediction_horizon
        self.congestion_threshold = congestion_threshold
        
        # Lịch sử tắc nghẽn để dự đoán tắc nghẽn tương lai
        self.congestion_history = [np.zeros(self.num_shards) for _ in range(prediction_horizon)]
        
        # Cache đường dẫn
        self.path_cache = {}
        
        # Xây dựng đồ thị shard level từ network
        self.shard_graph = self._build_shard_graph()
    
    def _build_shard_graph(self) -> nx.Graph:
        """
        Xây dựng đồ thị cấp shard từ đồ thị mạng.
        
        Returns:
            nx.Graph: Đồ thị cấp shard
        """
        shard_graph = nx.Graph()
        
        # Thêm các shard như là node
        for shard_id in range(self.num_shards):
            shard_graph.add_node(shard_id, 
                                congestion=0.0,
                                trust_score=0.0)
        
        # Tìm các kết nối giữa các shard và tính toán độ trễ/băng thông trung bình
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                cross_shard_edges = []
                
                # Tìm tất cả các cạnh kết nối node giữa hai shard
                for node_i in self.shards[i]:
                    for node_j in self.shards[j]:
                        if self.network.has_edge(node_i, node_j):
                            cross_shard_edges.append((node_i, node_j))
                
                if cross_shard_edges:
                    # Tính độ trễ và băng thông trung bình của các kết nối
                    avg_latency = np.mean([self.network.edges[u, v]['latency'] for u, v in cross_shard_edges])
                    avg_bandwidth = np.mean([self.network.edges[u, v]['bandwidth'] for u, v in cross_shard_edges])
                    
                    # Thêm cạnh giữa hai shard với độ trễ và băng thông trung bình
                    shard_graph.add_edge(i, j, 
                                        latency=avg_latency,
                                        bandwidth=avg_bandwidth,
                                        connection_count=len(cross_shard_edges))
        
        return shard_graph
    
    def update_network_state(self, shard_congestion: np.ndarray, node_trust_scores: Dict[int, float]):
        """
        Cập nhật trạng thái mạng với dữ liệu mới.
        
        Args:
            shard_congestion: Mảng chứa mức độ tắc nghẽn của từng shard
            node_trust_scores: Dictionary ánh xạ từ node ID đến điểm tin cậy
        """
        # Cập nhật lịch sử tắc nghẽn
        self.congestion_history.pop(0)
        self.congestion_history.append(shard_congestion.copy())
        
        # Cập nhật mức độ tắc nghẽn hiện tại cho mỗi shard trong shard_graph
        for shard_id in range(self.num_shards):
            self.shard_graph.nodes[shard_id]['congestion'] = shard_congestion[shard_id]
            
            # Tính điểm tin cậy trung bình cho shard
            shard_nodes = self.shards[shard_id]
            avg_trust = np.mean([node_trust_scores.get(node, 0.5) for node in shard_nodes])
            self.shard_graph.nodes[shard_id]['trust_score'] = avg_trust
        
        # Xóa cache đường dẫn khi trạng thái mạng thay đổi
        self.path_cache = {}
    
    def _predict_congestion(self, shard_id: int) -> float:
        """
        Dự đoán mức độ tắc nghẽn tương lai của một shard dựa trên lịch sử.
        
        Args:
            shard_id: ID của shard cần dự đoán
            
        Returns:
            float: Mức độ tắc nghẽn dự đoán
        """
        # Lấy lịch sử tắc nghẽn cho shard cụ thể
        congestion_values = [history[shard_id] for history in self.congestion_history]
        
        # Tính trọng số giảm dần theo thời gian (càng gần hiện tại càng quan trọng)
        weights = np.exp(np.linspace(0, 1, len(congestion_values)))
        weights = weights / np.sum(weights)
        
        # Tính tắc nghẽn dự đoán là trung bình có trọng số
        predicted_congestion = np.sum(weights * congestion_values)
        
        # Điều chỉnh dự đoán dựa trên xu hướng hiện tại
        if len(congestion_values) >= 2:
            trend = congestion_values[-1] - congestion_values[0]
            predicted_congestion += 0.2 * trend  # Thêm một phần của xu hướng
        
        # Đảm bảo giá trị nằm trong khoảng [0, 1]
        return np.clip(predicted_congestion, 0.0, 1.0)
    
    def _calculate_path_cost(self, path: List[int], transaction: Dict[str, Any]) -> float:
        """
        Tính chi phí đường dẫn dựa trên các yếu tố hiệu suất.
        
        Args:
            path: Đường dẫn là danh sách các shard ID
            transaction: Giao dịch cần định tuyến
            
        Returns:
            float: Chi phí tổng hợp của đường dẫn
        """
        if len(path) < 2:
            return 0.0
        
        total_latency = 0.0
        total_energy = 0.0
        total_congestion = 0.0
        total_trust = 0.0
        
        for i in range(len(path) - 1):
            shard_from = path[i]
            shard_to = path[i + 1]
            
            # Nếu không có kết nối trực tiếp giữa hai shard, trả về chi phí cao
            if not self.shard_graph.has_edge(shard_from, shard_to):
                return float('inf')
            
            # Lấy thông tin cạnh
            edge_data = self.shard_graph.edges[shard_from, shard_to]
            
            # Tính độ trễ của cạnh
            latency = edge_data['latency']
            total_latency += latency
            
            # Tính năng lượng tiêu thụ (dựa trên độ trễ và băng thông)
            # Giả định: Tiêu thụ năng lượng tỷ lệ thuận với độ trễ và tỷ lệ nghịch với băng thông
            energy = latency * (1.0 / edge_data['bandwidth']) * 10.0  # Hệ số 10.0 để chuẩn hóa
            total_energy += energy
            
            # Dự đoán tắc nghẽn của shard đích
            predicted_congestion = self._predict_congestion(shard_to)
            total_congestion += predicted_congestion
            
            # Lấy điểm tin cậy trung bình của shard đích
            trust_score = self.shard_graph.nodes[shard_to]['trust_score']
            total_trust += trust_score
        
        # Tính chi phí tổng hợp dựa trên các trọng số
        # Chi phí thấp hơn = đường dẫn tốt hơn
        cost = (self.congestion_weight * total_congestion / (len(path) - 1) + 
                self.latency_weight * total_latency / 100.0 +  # Chuẩn hóa độ trễ (giả sử tối đa 100ms)
                self.energy_weight * total_energy / 20.0 -     # Chuẩn hóa năng lượng (giả sử tối đa 20 đơn vị)
                self.trust_weight * total_trust / (len(path) - 1))  # Điểm tin cậy cao = chi phí thấp
        
        return cost
    
    def _dijkstra(self, source_shard: int, dest_shard: int, transaction: Dict[str, Any]) -> List[int]:
        """
        Thuật toán Dijkstra sửa đổi để tìm đường dẫn tối ưu giữa các shard.
        
        Args:
            source_shard: Shard nguồn
            dest_shard: Shard đích
            transaction: Giao dịch cần định tuyến
            
        Returns:
            List[int]: Đường dẫn tối ưu là danh sách các shard ID
        """
        # Kiểm tra cache
        cache_key = (source_shard, dest_shard, transaction['value'])
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Khởi tạo
        distances = {shard: float('inf') for shard in range(self.num_shards)}
        distances[source_shard] = 0
        previous = {shard: None for shard in range(self.num_shards)}
        priority_queue = [(0, source_shard)]
        
        while priority_queue:
            current_distance, current_shard = heapq.heappop(priority_queue)
            
            # Nếu đã đến đích, thoát khỏi vòng lặp
            if current_shard == dest_shard:
                break
            
            # Nếu khoảng cách hiện tại lớn hơn khoảng cách đã biết, bỏ qua
            if current_distance > distances[current_shard]:
                continue
            
            # Duyệt các shard kề
            for neighbor in self.shard_graph.neighbors(current_shard):
                # Xây dựng đường dẫn tạm thời đến neighbor
                temp_path = self._reconstruct_path(previous, current_shard)
                temp_path.append(neighbor)
                
                # Tính chi phí đường dẫn
                path_cost = self._calculate_path_cost(temp_path, transaction)
                
                # Nếu tìm thấy đường dẫn tốt hơn
                if path_cost < distances[neighbor]:
                    distances[neighbor] = path_cost
                    previous[neighbor] = current_shard
                    heapq.heappush(priority_queue, (path_cost, neighbor))
        
        # Xây dựng đường dẫn từ source đến dest
        path = self._reconstruct_path(previous, dest_shard)
        
        # Lưu vào cache
        self.path_cache[cache_key] = path
        
        return path
    
    def _reconstruct_path(self, previous: Dict[int, int], end: int) -> List[int]:
        """
        Xây dựng đường dẫn từ dict previous.
        
        Args:
            previous: Dictionary ánh xạ từ node đến node trước đó trong đường dẫn
            end: Node cuối cùng trong đường dẫn
            
        Returns:
            List[int]: Đường dẫn hoàn chỉnh
        """
        path = []
        current = end
        
        while current is not None:
            path.append(current)
            current = previous[current]
        
        # Đảo ngược đường dẫn để được từ nguồn đến đích
        return path[::-1]
    
    def find_optimal_path(self, 
                         transaction: Dict[str, Any], 
                         source_shard: int, 
                         destination_shard: int) -> List[int]:
        """
        Tìm đường dẫn tối ưu cho một giao dịch giữa hai shard.
        
        Args:
            transaction: Giao dịch cần định tuyến
            source_shard: Shard nguồn
            destination_shard: Shard đích
            
        Returns:
            List[int]: Đường dẫn tối ưu là danh sách các shard ID
        """
        # Nếu source và destination giống nhau, trả về đường dẫn một shard
        if source_shard == destination_shard:
            return [source_shard]
        
        # Tìm đường dẫn tối ưu bằng thuật toán Dijkstra sửa đổi
        return self._dijkstra(source_shard, destination_shard, transaction)
    
    def detect_congestion_hotspots(self) -> List[int]:
        """
        Phát hiện các điểm nóng tắc nghẽn trong mạng.
        
        Returns:
            List[int]: Danh sách các shard ID đang bị tắc nghẽn
        """
        hotspots = []
        
        for shard_id in range(self.num_shards):
            # Dự đoán tắc nghẽn
            predicted_congestion = self._predict_congestion(shard_id)
            
            # Nếu tắc nghẽn dự đoán vượt ngưỡng, coi là điểm nóng
            if predicted_congestion > self.congestion_threshold:
                hotspots.append(shard_id)
        
        return hotspots
    
    def find_optimal_paths_for_transactions(self, transaction_pool: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """
        Tìm đường dẫn tối ưu cho nhiều giao dịch cùng lúc.
        
        Args:
            transaction_pool: Danh sách các giao dịch cần định tuyến
            
        Returns:
            Dict[int, List[int]]: Dictionary ánh xạ từ transaction ID đến đường dẫn tối ưu
        """
        routes = {}
        
        for tx in transaction_pool:
            if tx['status'] == 'pending':
                # Lấy thông tin shard nguồn và đích
                source_shard = tx['source_shard']
                destination_shard = tx['destination_shard']
                
                # Tìm đường dẫn tối ưu
                path = self.find_optimal_path(tx, source_shard, destination_shard)
                
                # Lưu đường dẫn vào kết quả
                routes[tx['id']] = path
                
        return routes
    
    def optimize_routing_weights(self, 
                               recent_metrics: Dict[str, List[float]], 
                               target_latency: float = 0.0, 
                               target_energy: float = 0.0):
        """
        Tối ưu hóa các trọng số định tuyến dựa trên các metrics gần đây và mục tiêu.
        
        Args:
            recent_metrics: Dictionary chứa các metrics hiệu suất gần đây
            target_latency: Mục tiêu độ trễ (0.0 = không giới hạn)
            target_energy: Mục tiêu tiêu thụ năng lượng (0.0 = không giới hạn)
        """
        # Nếu độ trễ gần đây cao và mục tiêu latency > 0
        if target_latency > 0 and 'latency' in recent_metrics:
            avg_latency = np.mean(recent_metrics['latency'])
            if avg_latency > target_latency:
                # Tăng trọng số cho độ trễ
                self.latency_weight = min(0.6, self.latency_weight * 1.2)
                
                # Giảm các trọng số khác để tổng = 1.0
                total_other = self.congestion_weight + self.energy_weight + self.trust_weight
                scale_factor = (1.0 - self.latency_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.energy_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # Nếu tiêu thụ năng lượng gần đây cao và mục tiêu energy > 0
        if target_energy > 0 and 'energy_consumption' in recent_metrics:
            avg_energy = np.mean(recent_metrics['energy_consumption'])
            if avg_energy > target_energy:
                # Tăng trọng số cho năng lượng
                self.energy_weight = min(0.5, self.energy_weight * 1.2)
                
                # Giảm các trọng số khác để tổng = 1.0
                total_other = self.congestion_weight + self.latency_weight + self.trust_weight
                scale_factor = (1.0 - self.energy_weight) / total_other
                
                self.congestion_weight *= scale_factor
                self.latency_weight *= scale_factor
                self.trust_weight *= scale_factor
        
        # Đảm bảo tổng các trọng số = 1.0
        total_weight = self.congestion_weight + self.latency_weight + self.energy_weight + self.trust_weight
        if abs(total_weight - 1.0) > 1e-6:
            scale = 1.0 / total_weight
            self.congestion_weight *= scale
            self.latency_weight *= scale
            self.energy_weight *= scale
            self.trust_weight *= scale
        
        # Xóa cache khi thay đổi trọng số
        self.path_cache = {} 