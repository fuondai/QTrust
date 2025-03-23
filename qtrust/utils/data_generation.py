"""
Module chứa các hàm tạo dữ liệu mô phỏng cho hệ thống blockchain.
"""

import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Any, Optional

def generate_network_topology(num_nodes: int, 
                             avg_degree: float = 3.0,
                             p_rewire: float = 0.1,
                             seed: Optional[int] = None) -> nx.Graph:
    """
    Tạo mô hình mạng blockchain với topology Watts-Strogatz (small-world).
    
    Args:
        num_nodes: Số lượng node trong mạng
        avg_degree: Mức độ trung bình của mỗi node (số kết nối)
        p_rewire: Xác suất dây dẫn lại (rewiring probability)
        seed: Seed cho random generator
        
    Returns:
        nx.Graph: Đồ thị mạng blockchain
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Tính k dựa trên avg_degree (k phải là số chẵn trong Watts-Strogatz)
    k = max(2, int(avg_degree))
    if k % 2 == 1:
        k += 1
    
    # Tạo mạng small-world
    network = nx.watts_strogatz_graph(n=num_nodes, k=k, p=p_rewire, seed=seed)
    
    # Thêm thuộc tính cho các node và cạnh
    for node in network.nodes():
        # Băng thông và độ trễ ngẫu nhiên
        network.nodes[node]['bandwidth'] = np.random.uniform(10, 100)  # Mbps
        network.nodes[node]['processing_power'] = np.random.uniform(1, 10)  # Arbitrary units
        network.nodes[node]['storage'] = np.random.uniform(50, 500)  # GB
        network.nodes[node]['trust_score'] = np.random.uniform(0.5, 1.0)  # Initial trust
    
    for u, v in network.edges():
        # Độ trễ ngẫu nhiên giữa các node
        latency = np.random.uniform(5, 50)  # ms
        bandwidth = min(network.nodes[u]['bandwidth'], network.nodes[v]['bandwidth'])
        network[u][v]['latency'] = latency
        network[u][v]['bandwidth'] = bandwidth * 0.8  # Giảm 20% bandwidth do overhead
    
    return network

def assign_nodes_to_shards(network: nx.Graph, 
                          num_shards: int,
                          shard_method: str = 'random') -> List[List[int]]:
    """
    Phân chia các node vào các shard.
    
    Args:
        network: Đồ thị mạng blockchain
        num_shards: Số lượng shard cần tạo
        shard_method: Phương pháp phân chia ('random', 'balanced', 'spectral')
        
    Returns:
        List[List[int]]: Danh sách các shard và node trong mỗi shard
    """
    num_nodes = network.number_of_nodes()
    shards = [[] for _ in range(num_shards)]
    
    if shard_method == 'random':
        # Phân bổ ngẫu nhiên
        nodes = list(network.nodes())
        random.shuffle(nodes)
        
        for idx, node in enumerate(nodes):
            shard_id = idx % num_shards
            shards[shard_id].append(node)
            
    elif shard_method == 'balanced':
        # Phân bổ cân bằng dựa trên processing power
        nodes = list(network.nodes())
        # Sắp xếp node theo processing power (từ cao đến thấp)
        nodes.sort(key=lambda x: network.nodes[x]['processing_power'], reverse=True)
        
        # Phân bổ theo round-robin để cân bằng
        processing_power = [0] * num_shards
        
        for node in nodes:
            # Tìm shard có tổng processing power thấp nhất
            min_power_shard = processing_power.index(min(processing_power))
            shards[min_power_shard].append(node)
            processing_power[min_power_shard] += network.nodes[node]['processing_power']
            
    elif shard_method == 'spectral':
        # Sử dụng phương pháp spectral clustering để tạo các shard có kết nối mật thiết
        try:
            import sklearn.cluster as cluster
            
            # Tạo ma trận adjacency
            adj_matrix = nx.to_numpy_array(network)
            
            # Áp dụng spectral clustering
            spectral = cluster.SpectralClustering(n_clusters=num_shards, 
                                                affinity='precomputed',
                                                random_state=0)
            spectral.fit(adj_matrix)
            
            # Phân các node vào shard dựa trên kết quả clustering
            for node, label in enumerate(spectral.labels_):
                shards[label].append(node)
        except ImportError:
            print("sklearn không được cài đặt, sử dụng phương pháp phân chia ngẫu nhiên")
            return assign_nodes_to_shards(network, num_shards, 'random')
    else:
        raise ValueError(f"Phương pháp phân chia không hợp lệ: {shard_method}")
    
    return shards

def generate_transactions(num_transactions: int, 
                        num_nodes: int,
                        shards: List[List[int]],
                        value_range: Tuple[float, float] = (0.1, 100.0),
                        cross_shard_prob: float = 0.3,
                        seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Tạo dữ liệu giao dịch ngẫu nhiên.
    
    Args:
        num_transactions: Số lượng giao dịch cần tạo
        num_nodes: Tổng số node trong mạng
        shards: Danh sách các shard và node trong mỗi shard
        value_range: Khoảng giá trị giao dịch (min, max)
        cross_shard_prob: Xác suất giao dịch xuyên shard
        seed: Seed cho random generator
        
    Returns:
        List[Dict[str, Any]]: Danh sách các giao dịch
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    transactions = []
    
    # Map node to shard
    node_to_shard = {}
    for shard_id, nodes in enumerate(shards):
        for node in nodes:
            node_to_shard[node] = shard_id
    
    for i in range(num_transactions):
        # Tạo giao dịch mới
        transaction = {
            'id': i,
            'timestamp': np.random.uniform(0, 1000),  # Thời gian tương đối
            'value': np.random.uniform(value_range[0], value_range[1]),
            'gas_price': np.random.uniform(1, 10),
            'size': np.random.uniform(0.5, 2.0),  # KB
        }
        
        # Xác định source shard và node
        source_shard_id = np.random.randint(0, len(shards))
        source_node = random.choice(shards[source_shard_id])
        transaction['source'] = source_node
        
        # Xác định destination node
        is_cross_shard = random.random() < cross_shard_prob
        
        if is_cross_shard and len(shards) > 1:
            # Chọn shard khác với source
            dest_shard_candidates = [s for s in range(len(shards)) if s != source_shard_id]
            dest_shard_id = random.choice(dest_shard_candidates)
        else:
            # Cùng shard với source
            dest_shard_id = source_shard_id
        
        # Chọn node đích từ shard đích
        dest_node = random.choice(shards[dest_shard_id])
        while dest_node == source_node:  # Tránh self-transaction
            dest_node = random.choice(shards[dest_shard_id])
            
        transaction['destination'] = dest_node
        transaction['cross_shard'] = source_shard_id != dest_shard_id
        
        # Thêm vào danh sách
        transactions.append(transaction)
    
    # Sắp xếp theo timestamp
    transactions.sort(key=lambda x: x['timestamp'])
    
    return transactions

def generate_network_events(num_events: int, 
                          num_nodes: int,
                          duration: float = 1000.0,
                          seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Tạo các sự kiện mạng như node failure, network congestion, latency spikes.
    
    Args:
        num_events: Số lượng sự kiện cần tạo
        num_nodes: Tổng số node trong mạng
        duration: Thời gian mô phỏng (đơn vị tương đối)
        seed: Seed cho random generator
        
    Returns:
        List[Dict[str, Any]]: Danh sách các sự kiện mạng
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    events = []
    event_types = ['node_failure', 'congestion', 'latency_spike', 'bandwidth_drop']
    
    for i in range(num_events):
        # Thời gian sự kiện
        timestamp = np.random.uniform(0, duration)
        
        # Loại sự kiện
        event_type = random.choice(event_types)
        
        # Node hoặc cạnh bị ảnh hưởng
        affected_node = np.random.randint(0, num_nodes)
        
        # Tạo sự kiện
        event = {
            'id': i,
            'timestamp': timestamp,
            'type': event_type,
            'affected_node': affected_node,
            'duration': np.random.uniform(10, 100),  # Thời gian sự kiện kéo dài
        }
        
        # Thêm thông tin chi tiết dựa trên loại sự kiện
        if event_type == 'node_failure':
            event['severity'] = np.random.uniform(0.7, 1.0)  # Mức độ nghiêm trọng
            
        elif event_type == 'congestion':
            event['severity'] = np.random.uniform(0.3, 0.9)
            event['affected_links'] = []
            
            # Ảnh hưởng đến một số link liên quan
            num_affected_links = np.random.randint(1, 5)
            for _ in range(num_affected_links):
                target_node = np.random.randint(0, num_nodes)
                while target_node == affected_node:
                    target_node = np.random.randint(0, num_nodes)
                event['affected_links'].append((affected_node, target_node))
                
        elif event_type == 'latency_spike':
            event['multiplier'] = np.random.uniform(1.5, 5.0)  # Hệ số tăng độ trễ
            
        elif event_type == 'bandwidth_drop':
            event['reduction'] = np.random.uniform(0.3, 0.8)  # % giảm băng thông
        
        events.append(event)
    
    # Sắp xếp theo timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    return events

def generate_malicious_activities(num_activities: int,
                                 num_nodes: int,
                                 shards: List[List[int]],
                                 honest_node_prob: float = 0.9,
                                 seed: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[int, bool]]:
    """
    Tạo các hoạt động độc hại và xác định các node không trung thực.
    
    Args:
        num_activities: Số lượng hoạt động độc hại
        num_nodes: Tổng số node trong mạng
        shards: Danh sách các shard và node trong mỗi shard
        honest_node_prob: Xác suất một node là trung thực
        seed: Seed cho random generator
        
    Returns:
        Tuple[List[Dict[str, Any]], Dict[int, bool]]: 
            - Danh sách các hoạt động độc hại
            - Dict ánh xạ node ID đến trạng thái trung thực (True nếu trung thực)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Xác định node trung thực/không trung thực
    is_honest = {}
    for node in range(num_nodes):
        is_honest[node] = random.random() < honest_node_prob
    
    # Danh sách node không trung thực
    dishonest_nodes = [node for node, honest in is_honest.items() if not honest]
    
    if not dishonest_nodes:
        # Nếu không có node không trung thực, tạo một node độc hại
        dishonest_node = random.randint(0, num_nodes - 1)
        dishonest_nodes = [dishonest_node]
        is_honest[dishonest_node] = False
    
    # Tạo các hoạt động độc hại
    activities = []
    activity_types = [
        'double_spending', 'transaction_withholding', 'block_withholding',
        'sybil_attack', 'selfish_mining', 'eclipse_attack'
    ]
    
    for i in range(num_activities):
        # Chọn loại hoạt động
        activity_type = random.choice(activity_types)
        
        # Chọn node độc hại thực hiện hoạt động
        malicious_node = random.choice(dishonest_nodes)
        
        # Tạo hoạt động
        activity = {
            'id': i,
            'type': activity_type,
            'node': malicious_node,
            'timestamp': np.random.uniform(0, 1000),
            'severity': np.random.uniform(0.3, 1.0),
        }
        
        # Thêm thông tin chi tiết dựa trên loại hoạt động
        if activity_type == 'double_spending':
            activity['target_shard'] = random.randint(0, len(shards) - 1)
            activity['amount'] = np.random.uniform(10, 100)
            
        elif activity_type == 'transaction_withholding':
            activity['num_transactions'] = random.randint(1, 10)
            
        elif activity_type == 'block_withholding':
            activity['duration'] = np.random.uniform(10, 50)
            
        elif activity_type == 'sybil_attack':
            activity['fake_identities'] = random.randint(2, 5)
            
        elif activity_type == 'selfish_mining':
            activity['private_blocks'] = random.randint(1, 3)
            
        elif activity_type == 'eclipse_attack':
            victim_candidates = [node for node, honest in is_honest.items() if honest]
            if victim_candidates:
                activity['victim'] = random.choice(victim_candidates)
            else:
                activity['victim'] = random.randint(0, num_nodes - 1)
        
        activities.append(activity)
    
    # Sắp xếp theo timestamp
    activities.sort(key=lambda x: x['timestamp'])
    
    return activities, is_honest

def assign_trust_scores(num_nodes: int, 
                       is_honest: Dict[int, bool],
                       base_honest_score: float = 0.8,
                       honest_variance: float = 0.1,
                       base_dishonest_score: float = 0.4,
                       dishonest_variance: float = 0.2,
                       seed: Optional[int] = None) -> Dict[int, float]:
    """
    Khởi tạo điểm tin cậy cho các node dựa trên trạng thái trung thực.
    
    Args:
        num_nodes: Số lượng node trong mạng
        is_honest: Dictionary ánh xạ node ID đến trạng thái trung thực
        base_honest_score: Điểm cơ sở cho node trung thực
        honest_variance: Phương sai cho điểm cơ sở của node trung thực
        base_dishonest_score: Điểm cơ sở cho node không trung thực
        dishonest_variance: Phương sai cho điểm cơ sở của node không trung thực
        seed: Seed cho random generator
        
    Returns:
        Dict[int, float]: Dictionary ánh xạ node ID đến điểm tin cậy
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    trust_scores = {}
    
    for node in range(num_nodes):
        if is_honest.get(node, True):
            # Node trung thực
            score = base_honest_score + np.random.normal(0, honest_variance)
            # Giới hạn trong khoảng [0.5, 1.0]
            score = min(1.0, max(0.5, score))
        else:
            # Node không trung thực
            score = base_dishonest_score + np.random.normal(0, dishonest_variance)
            # Giới hạn trong khoảng [0.1, 0.7]
            score = min(0.7, max(0.1, score))
        
        trust_scores[node] = score
    
    return trust_scores 