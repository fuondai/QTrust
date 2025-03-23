import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque

class HTDCMNode:
    """
    Đại diện cho thông tin tin cậy của một node trong hệ thống.
    """
    def __init__(self, node_id: int, shard_id: int, initial_trust: float = 0.7):
        """
        Khởi tạo thông tin tin cậy cho một node.
        
        Args:
            node_id: ID của node
            shard_id: ID của shard mà node thuộc về
            initial_trust: Điểm tin cậy ban đầu (0.0-1.0)
        """
        self.node_id = node_id
        self.shard_id = shard_id
        self.trust_score = initial_trust
        
        # Lưu trữ lịch sử hoạt động
        self.successful_txs = 0
        self.failed_txs = 0
        self.malicious_activities = 0
        self.response_times = []
        
        # Các tham số cho việc tính toán tin cậy
        self.alpha = 0.8  # Trọng số cho lịch sử tin cậy
        self.beta = 0.2   # Trọng số cho đánh giá mới
        
        # Lưu trữ đánh giá từ các node khác
        self.peer_ratings = defaultdict(lambda: 0.5)  # Node ID -> Rating
        
        # Lịch sử hành vi
        self.activity_history = deque(maxlen=100)  # Lưu trữ 100 hoạt động gần nhất
    
    def update_trust_score(self, new_rating: float):
        """
        Cập nhật điểm tin cậy dựa trên đánh giá mới.
        
        Args:
            new_rating: Đánh giá mới (0.0-1.0)
        """
        # Cập nhật điểm tin cậy theo hàm trung bình có trọng số
        self.trust_score = self.alpha * self.trust_score + self.beta * new_rating
        
        # Đảm bảo điểm tin cậy nằm trong khoảng [0.0, 1.0]
        self.trust_score = max(0.0, min(1.0, self.trust_score))
    
    def record_transaction_result(self, success: bool, response_time: float, is_validator: bool):
        """
        Ghi lại kết quả một giao dịch mà node tham gia.
        
        Args:
            success: Giao dịch thành công hay thất bại
            response_time: Thời gian phản hồi (ms)
            is_validator: Node có là validator cho giao dịch hay không
        """
        if success:
            self.successful_txs += 1
            self.activity_history.append(('success', response_time, is_validator))
        else:
            self.failed_txs += 1
            self.activity_history.append(('fail', response_time, is_validator))
        
        self.response_times.append(response_time)
        
        # Giới hạn số lượng phản hồi lưu trữ
        if len(self.response_times) > 100:
            self.response_times.pop(0)
    
    def record_peer_rating(self, peer_id: int, rating: float):
        """
        Ghi lại đánh giá từ một node khác.
        
        Args:
            peer_id: ID của node đánh giá
            rating: Đánh giá (0.0-1.0)
        """
        self.peer_ratings[peer_id] = rating
    
    def record_malicious_activity(self, activity_type: str):
        """
        Ghi lại một hoạt động độc hại được phát hiện.
        
        Args:
            activity_type: Loại hoạt động độc hại
        """
        self.malicious_activities += 1
        self.activity_history.append(('malicious', activity_type, True))
        
        # Áp dụng hình phạt nghiêm khắc cho hoạt động độc hại
        self.update_trust_score(0.0)
    
    def get_average_response_time(self) -> float:
        """
        Lấy thời gian phản hồi trung bình gần đây.
        
        Returns:
            float: Thời gian phản hồi trung bình
        """
        if not self.response_times:
            return 0.0
        return np.mean(self.response_times)
    
    def get_success_rate(self) -> float:
        """
        Lấy tỷ lệ thành công của các giao dịch.
        
        Returns:
            float: Tỷ lệ thành công
        """
        total = self.successful_txs + self.failed_txs
        if total == 0:
            return 0.0
        return self.successful_txs / total
    
    def get_peer_trust(self) -> float:
        """
        Lấy điểm tin cậy trung bình từ các node khác.
        
        Returns:
            float: Điểm tin cậy trung bình từ các peers
        """
        if not self.peer_ratings:
            return 0.5
        return np.mean(list(self.peer_ratings.values()))

class HTDCM:
    """
    Hierarchical Trust-based Data Center Mechanism (HTDCM).
    Cơ chế đánh giá tin cậy đa cấp cho mạng blockchain.
    """
    
    def __init__(self, 
                 network: nx.Graph,
                 shards: List[List[int]],
                 tx_success_weight: float = 0.4,
                 response_time_weight: float = 0.2,
                 peer_rating_weight: float = 0.3,
                 history_weight: float = 0.1,
                 malicious_threshold: float = 0.3,
                 suspicious_pattern_window: int = 10):
        """
        Khởi tạo hệ thống đánh giá tin cậy HTDCM.
        
        Args:
            network: Đồ thị mạng blockchain
            shards: Danh sách các shard và node trong mỗi shard
            tx_success_weight: Trọng số cho tỷ lệ giao dịch thành công
            response_time_weight: Trọng số cho thời gian phản hồi
            peer_rating_weight: Trọng số cho đánh giá từ các node khác
            history_weight: Trọng số cho lịch sử hành vi
            malicious_threshold: Ngưỡng điểm tin cậy để coi là độc hại
            suspicious_pattern_window: Kích thước cửa sổ để phát hiện mẫu đáng ngờ
        """
        self.network = network
        self.shards = shards
        self.num_shards = len(shards)
        
        # Trọng số cho các yếu tố khác nhau trong đánh giá tin cậy
        self.tx_success_weight = tx_success_weight
        self.response_time_weight = response_time_weight
        self.peer_rating_weight = peer_rating_weight
        self.history_weight = history_weight
        
        # Ngưỡng và tham số phát hiện độc hại
        self.malicious_threshold = malicious_threshold
        self.suspicious_pattern_window = suspicious_pattern_window
        
        # Điểm tin cậy của các shard
        self.shard_trust_scores = np.ones(self.num_shards) * 0.7
        
        # Khởi tạo thông tin tin cậy cho mỗi node
        self.nodes = {}
        for shard_id, shard_nodes in enumerate(shards):
            for node_id in shard_nodes:
                initial_trust = self.network.nodes[node_id].get('trust_score', 0.7)
                self.nodes[node_id] = HTDCMNode(node_id, shard_id, initial_trust)
        
        # Lịch sử đánh giá toàn cục
        self.global_ratings_history = []
    
    def update_node_trust(self, 
                        node_id: int, 
                        tx_success: bool, 
                        response_time: float, 
                        is_validator: bool):
        """
        Cập nhật điểm tin cậy của một node dựa trên kết quả giao dịch.
        
        Args:
            node_id: ID của node
            tx_success: Giao dịch thành công hay không
            response_time: Thời gian phản hồi (ms)
            is_validator: Node có phải là validator hay không
        """
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Ghi lại kết quả giao dịch
        node.record_transaction_result(tx_success, response_time, is_validator)
        
        # Tính toán đánh giá mới
        rating = self._calculate_node_rating(node)
        
        # Cập nhật điểm tin cậy
        node.update_trust_score(rating)
        
        # Cập nhật điểm tin cậy trong mạng
        self.network.nodes[node_id]['trust_score'] = node.trust_score
        
        # Cập nhật điểm tin cậy của shard
        self._update_shard_trust(node.shard_id)
        
        # Lưu lịch sử đánh giá toàn cục
        self.global_ratings_history.append((node_id, node.trust_score))
        
        # Phát hiện hoạt động đáng ngờ
        self._detect_suspicious_behavior(node_id)
    
    def _calculate_node_rating(self, node: HTDCMNode) -> float:
        """
        Tính toán đánh giá cho một node dựa trên nhiều yếu tố.
        
        Args:
            node: Đối tượng HTDCMNode
            
        Returns:
            float: Đánh giá (0.0-1.0)
        """
        # Tỷ lệ giao dịch thành công
        success_rate = node.get_success_rate()
        
        # Thời gian phản hồi (chuẩn hóa: thấp hơn = tốt hơn)
        avg_response_time = node.get_average_response_time()
        normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)  # Giả sử 100ms là tối đa
        
        # Đánh giá từ các peer
        peer_trust = node.get_peer_trust()
        
        # Điểm lịch sử (các hoạt động độc hại trước đây)
        history_score = 1.0 - min(1.0, node.malicious_activities / 10.0)  # Giả sử 10 hoạt động là tối đa
        
        # Tính điểm tổng hợp
        rating = (self.tx_success_weight * success_rate +
                 self.response_time_weight * normalized_response_time +
                 self.peer_rating_weight * peer_trust +
                 self.history_weight * history_score)
        
        return rating
    
    def _update_shard_trust(self, shard_id: int):
        """
        Cập nhật điểm tin cậy cho một shard dựa trên các node trong shard đó.
        
        Args:
            shard_id: ID của shard
        """
        # Lấy tất cả các node trong shard
        shard_nodes = [self.nodes[node_id] for node_id in self.shards[shard_id]]
        
        # Tính điểm tin cậy trung bình
        avg_trust = np.mean([node.trust_score for node in shard_nodes])
        
        # Cập nhật điểm tin cậy của shard
        self.shard_trust_scores[shard_id] = avg_trust
    
    def _detect_suspicious_behavior(self, node_id: int):
        """
        Phát hiện các hành vi đáng ngờ của một node.
        
        Args:
            node_id: ID của node cần kiểm tra
        """
        node = self.nodes[node_id]
        
        # Nếu điểm tin cậy dưới ngưỡng, coi là độc hại
        if node.trust_score < self.malicious_threshold:
            node.record_malicious_activity('low_trust')
            return True
        
        # Phát hiện mẫu đáng ngờ trong lịch sử hoạt động
        if len(node.activity_history) >= self.suspicious_pattern_window:
            recent_activities = list(node.activity_history)[-self.suspicious_pattern_window:]
            
            # Kiểm tra nếu node liên tục thất bại trong các giao dịch
            fail_count = sum(1 for act in recent_activities if act[0] == 'fail')
            if fail_count >= self.suspicious_pattern_window * 0.8:
                node.record_malicious_activity('consistent_failure')
                return True
            
            # Kiểm tra nếu node có thời gian phản hồi bất thường
            response_times = [act[1] for act in recent_activities if isinstance(act[1], (int, float))]
            if response_times and np.std(response_times) > 3 * np.mean(response_times):
                node.record_malicious_activity('erratic_response_time')
                return True
        
        return False
    
    def rate_peers(self, observer_id: int, transactions: List[Dict[str, Any]]):
        """
        Cho phép một node đánh giá các node khác dựa trên các giao dịch chung.
        
        Args:
            observer_id: ID của node quan sát
            transactions: Danh sách các giao dịch mà node quan sát tham gia
        """
        if observer_id not in self.nodes:
            return
        
        # Điểm tin cậy của node quan sát
        observer_trust = self.nodes[observer_id].trust_score
        
        # Tạo dictionary để theo dõi các node được quan sát
        observed_nodes = defaultdict(list)
        
        for tx in transactions:
            # Lấy danh sách các node tham gia trong giao dịch (ngoài observer)
            participant_nodes = []
            if 'source_node' in tx and tx['source_node'] != observer_id:
                participant_nodes.append(tx['source_node'])
            if 'destination_node' in tx and tx['destination_node'] != observer_id:
                participant_nodes.append(tx['destination_node'])
            
            # Thêm thông tin về sự tham gia của mỗi node trong giao dịch này
            for node_id in participant_nodes:
                if node_id in self.nodes:
                    observed_nodes[node_id].append({
                        'success': tx['status'] == 'completed',
                        'response_time': tx.get('completion_time', 0) - tx.get('created_at', 0)
                    })
        
        # Đánh giá từng node dựa trên hiệu suất trong các giao dịch chung
        for node_id, observations in observed_nodes.items():
            if not observations:
                continue
            
            # Tính tỷ lệ thành công và thời gian phản hồi trung bình
            success_rate = sum(1 for obs in observations if obs['success']) / len(observations)
            avg_response_time = np.mean([obs['response_time'] for obs in observations])
            
            # Chuẩn hóa thời gian phản hồi
            normalized_response_time = 1.0 - min(1.0, avg_response_time / 100.0)
            
            # Tính rating tổng hợp
            rating = 0.7 * success_rate + 0.3 * normalized_response_time
            
            # Lưu đánh giá vào node được quan sát
            self.nodes[node_id].record_peer_rating(observer_id, rating)
            
            # Cập nhật điểm tin cậy của node được quan sát (với trọng số thấp hơn)
            peer_influence = min(0.1, observer_trust * 0.2)  # Trọng số của đánh giá từ peer
            self.nodes[node_id].update_trust_score(
                self.nodes[node_id].trust_score * (1 - peer_influence) + rating * peer_influence
            )
    
    def get_node_trust_scores(self) -> Dict[int, float]:
        """
        Lấy danh sách điểm tin cậy của tất cả các node.
        
        Returns:
            Dict[int, float]: Dictionary ánh xạ node ID đến điểm tin cậy
        """
        return {node_id: node.trust_score for node_id, node in self.nodes.items()}
    
    def get_shard_trust_scores(self) -> np.ndarray:
        """
        Lấy danh sách điểm tin cậy của tất cả các shard.
        
        Returns:
            np.ndarray: Mảng chứa điểm tin cậy của các shard
        """
        return self.shard_trust_scores
    
    def identify_malicious_nodes(self) -> List[int]:
        """
        Nhận diện các node độc hại trong mạng.
        
        Returns:
            List[int]: Danh sách ID của các node được xác định là độc hại
        """
        return [node_id for node_id, node in self.nodes.items() 
                if node.trust_score < self.malicious_threshold]
    
    def recommend_trusted_validators(self, shard_id: int, count: int = 3) -> List[int]:
        """
        Đề xuất các validator đáng tin cậy nhất cho một shard.
        
        Args:
            shard_id: ID của shard
            count: Số lượng validator cần đề xuất
            
        Returns:
            List[int]: Danh sách ID của các node được đề xuất làm validator
        """
        # Lấy tất cả các node trong shard đã cho
        shard_node_ids = self.shards[shard_id]
        
        # Sắp xếp các node theo điểm tin cậy giảm dần
        trusted_nodes = sorted(
            [(node_id, self.nodes[node_id].trust_score) for node_id in shard_node_ids],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Trả về danh sách ID của các node tin cậy nhất
        return [node_id for node_id, _ in trusted_nodes[:count]] 