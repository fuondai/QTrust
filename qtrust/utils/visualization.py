"""
Module chứa các hàm hiển thị trực quan mạng blockchain.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional

def plot_blockchain_network(network: nx.Graph, 
                           shards: List[List[int]], 
                           trust_scores: Optional[Dict[int, float]] = None,
                           title: str = "Blockchain Network",
                           figsize: Tuple[int, int] = (12, 10),
                           save_path: Optional[str] = None):
    """
    Vẽ mạng blockchain với các shard và node.
    
    Args:
        network: Đồ thị mạng blockchain
        shards: Danh sách các shard và node trong mỗi shard
        trust_scores: Điểm tin cậy của các node (nếu có)
        title: Tiêu đề của đồ thị
        figsize: Kích thước hình vẽ
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    # Xác định màu cho mỗi shard
    colors = plt.cm.jet(np.linspace(0, 1, len(shards)))
    
    # Xác định vị trí cho các node
    pos = nx.spring_layout(network, seed=42)
    
    # Vẽ các node và cạnh
    for shard_id, shard_nodes in enumerate(shards):
        # Vẽ các node trong shard này
        node_size = 300
        if trust_scores is not None:
            # Kích thước node dựa trên điểm tin cậy
            node_size = [trust_scores.get(node_id, 0.5) * 500 for node_id in shard_nodes]
        
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=shard_nodes,
            node_color=[colors[shard_id]] * len(shard_nodes),
            node_size=node_size,
            alpha=0.8,
            label=f"Shard {shard_id}"
        )
    
    # Vẽ các cạnh trong shard (màu đậm hơn)
    intra_shard_edges = []
    for shard_nodes in shards:
        for i, node1 in enumerate(shard_nodes):
            for node2 in shard_nodes[i+1:]:
                if network.has_edge(node1, node2):
                    intra_shard_edges.append((node1, node2))
    
    nx.draw_networkx_edges(
        network, pos,
        edgelist=intra_shard_edges,
        width=2.0,
        alpha=0.7
    )
    
    # Vẽ các cạnh xuyên shard (màu nhạt hơn, đường đứt)
    cross_shard_edges = []
    for edge in network.edges():
        if edge not in intra_shard_edges and (edge[1], edge[0]) not in intra_shard_edges:
            cross_shard_edges.append(edge)
    
    nx.draw_networkx_edges(
        network, pos,
        edgelist=cross_shard_edges,
        width=1.0,
        alpha=0.3,
        style='dashed'
    )
    
    # Thêm nhãn cho các node
    nx.draw_networkx_labels(
        network, pos,
        font_size=8,
        font_color='black'
    )
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_transaction_flow(network: nx.Graph, 
                         transactions: List[Dict[str, Any]], 
                         paths: Dict[int, List[int]],
                         title: str = "Transaction Flow",
                         figsize: Tuple[int, int] = (12, 10),
                         save_path: Optional[str] = None):
    """
    Vẽ luồng giao dịch trên mạng blockchain.
    
    Args:
        network: Đồ thị mạng blockchain
        transactions: Danh sách các giao dịch
        paths: Dictionary ánh xạ transaction ID đến đường đi
        title: Tiêu đề của đồ thị
        figsize: Kích thước hình vẽ
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    # Xác định vị trí cho các node
    pos = nx.spring_layout(network, seed=42)
    
    # Vẽ đồ thị mạng nền
    nx.draw_networkx_nodes(
        network, pos,
        node_size=300,
        node_color='lightgray',
        alpha=0.6
    )
    
    nx.draw_networkx_edges(
        network, pos,
        width=1.0,
        alpha=0.2
    )
    
    nx.draw_networkx_labels(
        network, pos,
        font_size=8,
        font_color='black'
    )
    
    # Vẽ các luồng giao dịch
    colors = plt.cm.rainbow(np.linspace(0, 1, len(transactions)))
    
    for idx, (tx_id, path) in enumerate(paths.items()):
        if len(path) < 2:
            continue
        
        # Tìm thông tin giao dịch
        tx_info = next((tx for tx in transactions if tx.get('id') == tx_id), None)
        if not tx_info:
            continue
        
        # Tạo đường đi
        edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        
        # Vẽ đường đi
        nx.draw_networkx_edges(
            network, pos,
            edgelist=edges,
            width=2.0,
            alpha=0.8,
            edge_color=[colors[idx % len(colors)]] * len(edges),
            arrows=True,
            connectionstyle='arc3,rad=0.1'
        )
        
        # Vẽ node nguồn và đích với màu đặc biệt
        source_node = path[0]
        dest_node = path[-1]
        
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=[source_node],
            node_color='green',
            node_size=400,
            alpha=0.8
        )
        
        nx.draw_networkx_nodes(
            network, pos,
            nodelist=[dest_node],
            node_color='red',
            node_size=400,
            alpha=0.8
        )
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_shard_graph(network: nx.Graph,
                    shards: List[List[int]],
                    shard_metrics: Dict[int, Dict[str, float]],
                    metric_name: str,
                    title: str = "Shard Performance",
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None):
    """
    Vẽ đồ thị hiệu suất của các shard.
    
    Args:
        network: Đồ thị mạng blockchain
        shards: Danh sách các shard và node trong mỗi shard
        shard_metrics: Dictionary ánh xạ shard ID đến dictionary metrics
        metric_name: Tên của chỉ số cần hiển thị
        title: Tiêu đề của đồ thị
        figsize: Kích thước hình vẽ
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    # Tạo đồ thị shard level
    shard_graph = nx.Graph()
    
    # Thêm node cho mỗi shard
    for shard_id in range(len(shards)):
        shard_graph.add_node(shard_id)
    
    # Thêm cạnh giữa các shard nếu có kết nối
    for shard_i in range(len(shards)):
        for shard_j in range(shard_i + 1, len(shards)):
            # Kiểm tra xem có cạnh nào giữa 2 shard không
            has_connection = False
            for node_i in shards[shard_i]:
                for node_j in shards[shard_j]:
                    if network.has_edge(node_i, node_j):
                        has_connection = True
                        break
                if has_connection:
                    break
            
            if has_connection:
                shard_graph.add_edge(shard_i, shard_j)
    
    # Xác định vị trí cho các shard
    pos = nx.circular_layout(shard_graph)
    
    # Vẽ các node shard với kích thước dựa trên metric
    node_sizes = []
    node_colors = []
    labels = {}
    
    min_val = min([metrics.get(metric_name, 0) for metrics in shard_metrics.values()])
    max_val = max([metrics.get(metric_name, 0) for metrics in shard_metrics.values()])
    if min_val == max_val:
        max_val = min_val + 1  # Để tránh chia cho 0
    
    for shard_id in range(len(shards)):
        metric_value = shard_metrics.get(shard_id, {}).get(metric_name, 0)
        
        # Chuẩn hóa kích thước
        size = 500 + (metric_value - min_val) / (max_val - min_val) * 1500
        node_sizes.append(size)
        
        # Chuẩn hóa màu sắc (xanh tốt, đỏ xấu)
        # Đảo ngược nếu metric là latency hoặc energy vì giá trị thấp là tốt
        if metric_name in ['latency', 'energy_consumption']:
            color_val = 1 - (metric_value - min_val) / (max_val - min_val)
        else:
            color_val = (metric_value - min_val) / (max_val - min_val)
        
        node_colors.append(plt.cm.RdYlGn(color_val))
        
        # Tạo nhãn
        labels[shard_id] = f"S{shard_id}\n{metric_value:.2f}"
    
    # Vẽ đồ thị
    nx.draw_networkx_nodes(
        shard_graph, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    
    nx.draw_networkx_edges(
        shard_graph, pos,
        width=2.0,
        alpha=0.5
    )
    
    nx.draw_networkx_labels(
        shard_graph, pos,
        labels=labels,
        font_size=10,
        font_color='black'
    )
    
    plt.title(f"{title}: {metric_name}")
    plt.axis('off')
    
    # Thêm thanh màu
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    
    if metric_name in ['latency', 'energy_consumption']:
        cbar.set_label('Low (good) to High (bad)')
    else:
        cbar.set_label('Low (bad) to High (good)')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_consensus_comparison(consensus_results: Dict[str, Dict[str, List[float]]],
                             figsize: Tuple[int, int] = (15, 10),
                             save_path: Optional[str] = None):
    """
    Vẽ đồ thị so sánh các giao thức đồng thuận.
    
    Args:
        consensus_results: Dictionary chứa kết quả của các giao thức đồng thuận
        figsize: Kích thước hình vẽ
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    metrics = ['latency', 'energy', 'success_rate']
    n_metrics = len(metrics)
    metric_labels = ['Latency (ms)', 'Energy Consumption', 'Success Rate']
    
    for i, metric in enumerate(metrics):
        plt.subplot(1, n_metrics, i+1)
        
        data = []
        labels = []
        
        for protocol, results in consensus_results.items():
            if metric in results:
                data.append(results[metric])
                labels.append(protocol)
        
        # Vẽ boxplot
        box = plt.boxplot(data, patch_artist=True, labels=labels)
        
        # Đặt màu cho các hộp
        colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
        for patch, color in zip(box['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
        
        plt.title(f'Comparison of {metric_labels[i]}')
        plt.ylabel(metric_labels[i])
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_learning_curve(rewards: List[float], 
                       avg_window: int = 20,
                       title: str = "Learning Curve",
                       figsize: Tuple[int, int] = (10, 6),
                       save_path: Optional[str] = None):
    """
    Vẽ đường cong học tập (learning curve) cho DQN Agent.
    
    Args:
        rewards: Danh sách phần thưởng từng episode
        avg_window: Kích thước cửa sổ cho đường trung bình động
        title: Tiêu đề của đồ thị
        figsize: Kích thước hình vẽ
        save_path: Đường dẫn để lưu đồ thị
    """
    plt.figure(figsize=figsize)
    
    episodes = range(1, len(rewards) + 1)
    
    # Vẽ phần thưởng gốc
    plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Episode Reward')
    
    # Tính và vẽ đường trung bình động
    if len(rewards) >= avg_window:
        avg_rewards = []
        for i in range(len(rewards) - avg_window + 1):
            avg_rewards.append(np.mean(rewards[i:i+avg_window]))
        
        plt.plot(range(avg_window, len(rewards) + 1), avg_rewards, 'r-', 
                 linewidth=2, label=f'Moving Average (window={avg_window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 