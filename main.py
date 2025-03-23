#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QTrust - Ứng dụng blockchain sharding tối ưu với Deep Reinforcement Learning

Tệp này là điểm vào chính cho việc chạy các mô phỏng QTrust.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import time
import random
from pathlib import Path

# Thêm thư mục hiện tại vào PYTHONPATH để đảm bảo các module có thể được import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from qtrust.simulation.blockchain_environment import BlockchainEnvironment
from qtrust.agents.dqn_agent import DQNAgent
from qtrust.consensus.adaptive_consensus import AdaptiveConsensus
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.trust.htdcm import HTDCM
from qtrust.federated.federated_learning import FederatedLearning, FederatedModel, FederatedClient
from qtrust.utils.metrics import (
    calculate_throughput, 
    calculate_latency_metrics,
    calculate_energy_efficiency,
    calculate_security_metrics,
    calculate_cross_shard_transaction_ratio,
    plot_performance_metrics,
    plot_comparison_charts
)
from qtrust.utils.data_generation import (
    generate_network_topology,
    assign_nodes_to_shards,
    generate_transactions
)

# Thiết lập ngẫu nhiên cho khả năng tái tạo
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    """Phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description='QTrust - Hệ thống Blockchain Sharding tối ưu với DRL')
    
    parser.add_argument('--num-shards', type=int, default=4, 
                        help='Số lượng shard trong mạng (mặc định: 4)')
    parser.add_argument('--nodes-per-shard', type=int, default=10, 
                        help='Số lượng node trong mỗi shard (mặc định: 10)')
    parser.add_argument('--episodes', type=int, default=500, 
                        help='Số lượng episode để huấn luyện (mặc định: 500)')
    parser.add_argument('--max-steps', type=int, default=1000, 
                        help='Số bước tối đa trong mỗi episode (mặc định: 1000)')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Kích thước batch cho mô hình DQN (mặc định: 64)')
    parser.add_argument('--hidden-size', type=int, default=128, 
                        help='Kích thước lớp ẩn của mô hình DQN (mặc định: 128)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Tốc độ học (mặc định: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Hệ số chiết khấu (mặc định: 0.99)')
    parser.add_argument('--epsilon-start', type=float, default=1.0, 
                        help='Giá trị epsilon khởi đầu (mặc định: 1.0)')
    parser.add_argument('--epsilon-end', type=float, default=0.01, 
                        help='Giá trị epsilon cuối cùng (mặc định: 0.01)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, 
                        help='Tốc độ giảm epsilon (mặc định: 0.995)')
    parser.add_argument('--memory-size', type=int, default=10000, 
                        help='Kích thước bộ nhớ replay (mặc định: 10000)')
    parser.add_argument('--target-update', type=int, default=10, 
                        help='Cập nhật mô hình target sau mỗi bao nhiêu episode (mặc định: 10)')
    parser.add_argument('--save-dir', type=str, default='models', 
                        help='Thư mục lưu các mô hình (mặc định: "models")')
    parser.add_argument('--log-interval', type=int, default=20, 
                        help='Ghi log sau mỗi bao nhiêu episode (mặc định: 20)')
    parser.add_argument('--enable-federated', action='store_true', 
                        help='Bật chế độ học liên hợp (Federated Learning)')
    parser.add_argument('--eval', action='store_true',
                        help='Chạy đánh giá trên mô hình đã huấn luyện')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Đường dẫn đến mô hình đã huấn luyện (cho đánh giá)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Thiết bị sử dụng cho học sâu (mặc định: auto)')
    
    return parser.parse_args()

def setup_environment(args):
    """Thiết lập môi trường blockchain."""
    print("Khởi tạo môi trường blockchain...")
    
    env = BlockchainEnvironment(
        num_shards=args.num_shards,
        num_nodes_per_shard=args.nodes_per_shard,
        max_transactions_per_step=50,
        max_steps=args.max_steps
    )
    
    return env

def setup_components(env, args):
    """Thiết lập các thành phần cho hệ thống QTrust."""
    print("Khởi tạo các thành phần của hệ thống QTrust...")
    
    # Cài đặt Multi-Agent Dynamic Routing
    router = MADRAPIDRouter(
        network=env.network,
        shards=env.shards,
        congestion_weight=0.4,
        latency_weight=0.3,
        energy_weight=0.2,
        trust_weight=0.1
    )
    
    # Cài đặt Adaptive Cross-Shard Consensus
    consensus = AdaptiveConsensus()
    
    # Cài đặt Hierarchical Trust-based Data Center Mechanism
    htdcm = HTDCM(
        network=env.network,
        shards=env.shards,
        tx_success_weight=0.4,
        response_time_weight=0.2,
        peer_rating_weight=0.3,
        history_weight=0.1
    )
    
    return router, consensus, htdcm

def setup_dqn_agent(env, args):
    """Thiết lập DQN Agent."""
    print("Khởi tạo DQN Agent...")
    
    # In kích thước không gian trạng thái của môi trường để debug
    print(f"Kích thước observation_space: {env.observation_space.shape}")
    print(f"Số lượng shard: {env.num_shards}")
    
    # Tạo state_space và action_space theo định dạng dictionary mà DQNAgent yêu cầu
    state_space = {"shape": env.observation_space.shape}
    action_space = {"shape": env.action_space.shape if hasattr(env.action_space, "shape") else [env.action_space.n]}
    
    if hasattr(env.action_space, 'nvec'):
        action_space["shape"] = list(env.action_space.nvec)
    
    agent = DQNAgent(
        state_space=state_space,
        action_space=action_space,
        num_shards=env.num_shards,  # Sử dụng env.num_shards thay vì args.num_shards
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.memory_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    return agent

def setup_federated_learning(env, args, htdcm):
    """Thiết lập hệ thống Federated Learning."""
    if not args.enable_federated:
        return None
    
    print("Khởi tạo hệ thống Federated Learning...")
    
    # Khởi tạo mô hình toàn cục
    input_size = env.observation_space.shape[0]
    hidden_size = args.hidden_size
    output_size = env.action_space.n
    
    global_model = FederatedModel(input_size, hidden_size, output_size)
    
    # Khởi tạo hệ thống Federated Learning
    fl_system = FederatedLearning(
        global_model=global_model,
        aggregation_method='fedtrust',
        client_selection_method='trust_based',
        min_clients_per_round=3,
        trust_threshold=0.5,
        device=args.device
    )
    
    # Tạo một client cho mỗi shard
    for shard_id in range(env.num_shards):
        # Lấy điểm tin cậy trung bình của shard
        shard_trust = htdcm.shard_trust_scores[shard_id]
        
        # Tạo client mới
        client = FederatedClient(
            client_id=shard_id,
            model=global_model,
            learning_rate=args.lr,
            local_epochs=5,
            batch_size=32,
            trust_score=shard_trust,
            device=args.device
        )
        
        # Thêm client vào hệ thống
        fl_system.add_client(client)
    
    return fl_system

def train_qtrust(env, agent, router, consensus, htdcm, fl_system, args):
    """Huấn luyện hệ thống QTrust."""
    print("Bắt đầu huấn luyện hệ thống QTrust...")
    
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Theo dõi hiệu suất
    episode_rewards = []
    avg_rewards = []
    transaction_throughputs = []
    latencies = []
    malicious_detections = []
    
    # Huấn luyện qua nhiều episode
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Lưu trữ dữ liệu cho federated learning
        shard_experiences = [[] for _ in range(env.num_shards)]
        
        # Chạy một episode
        while not done and steps < args.max_steps:
            # Chọn hành động từ DQN Agent
            action = agent.act(state)
            
            # Mô phỏng các hoạt động trong mạng blockchain
            
            # 1. Định tuyến các giao dịch
            transaction_routes = router.find_optimal_paths_for_transactions(env.transaction_pool)
            
            # 2. Thực hiện giao dịch với các giao thức đồng thuận thích ứng
            for tx in env.transaction_pool:
                # Chọn giao thức đồng thuận dựa trên giá trị giao dịch và tình trạng mạng
                trust_scores_dict = htdcm.get_node_trust_scores()
                protocol = consensus.select_protocol(
                    transaction_value=tx['value'],
                    network_congestion=env.get_congestion_level(),
                    trust_scores=list(trust_scores_dict.values())
                )
                
                # Thực hiện giao thức đồng thuận
                result, latency, energy = consensus.execute_consensus(
                    tx, env.network, protocol
                )
                
                # Ghi nhận kết quả vào hệ thống tin cậy
                for node_id in tx.get('validator_nodes', []):
                    htdcm.update_node_trust(
                        node_id=node_id,
                        tx_success=result,
                        response_time=latency,
                        is_validator=True
                    )
            
            # 3. Thực hiện bước trong môi trường với hành động đã chọn
            next_state, reward, done, info = env.step(action)
            
            # 4. Cập nhật Agent
            agent.step(state, action, reward, next_state, done)
            
            # Lưu trữ trải nghiệm cho từng shard
            for shard_id in range(env.num_shards):
                # Lấy các giao dịch liên quan đến shard này
                shard_txs = [tx for tx in env.transaction_pool 
                            if tx.get('shard_id') == shard_id or tx.get('destination_shard') == shard_id]
                
                if shard_txs:
                    # Lưu trữ trải nghiệm (state, action, reward, next_state)
                    shard_experience = (state, action, reward, next_state, done)
                    shard_experiences[shard_id].append(shard_experience)
            
            # Cập nhật cho bước tiếp theo
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Cập nhật trạng thái mạng cho router
            router.update_network_state(
                shard_congestion=env.get_congestion_data(),
                node_trust_scores=htdcm.get_node_trust_scores()
            )
        
        # Cập nhật target network nếu cần
        if episode % args.target_update == 0:
            agent.update_target_network()
        
        # Thực hiện Federated Learning nếu được bật
        if fl_system is not None and episode % 5 == 0:
            # Chuẩn bị dữ liệu huấn luyện cho mỗi client
            for shard_id in range(env.num_shards):
                if len(shard_experiences[shard_id]) > 0:
                    # Chuyển đổi các trải nghiệm thành dữ liệu huấn luyện
                    states = torch.FloatTensor([exp[0] for exp in shard_experiences[shard_id]])
                    actions = torch.LongTensor([[exp[1]] for exp in shard_experiences[shard_id]])
                    rewards = torch.FloatTensor([[exp[2]] for exp in shard_experiences[shard_id]])
                    
                    # Thiết lập dữ liệu cục bộ cho client
                    fl_system.clients[shard_id].set_local_data(
                        train_data=(states, actions),
                        val_data=None
                    )
                    
                    # Cập nhật điểm tin cậy cho client dựa trên điểm tin cậy của shard
                    fl_system.update_client_trust(
                        client_id=shard_id,
                        trust_score=htdcm.shard_trust_scores[shard_id]
                    )
            
            # Thực hiện một vòng huấn luyện Federated Learning
            round_metrics = fl_system.train_round(
                round_num=episode // 5,
                client_fraction=0.8
            )
            
            if round_metrics:
                print(f"Vòng Federated {episode//5}: "
                     f"Mất mát = {round_metrics['avg_train_loss']:.4f}")
        
        # Theo dõi hiệu suất
        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        avg_rewards.append(avg_reward)
        
        # Lưu các metrics
        transaction_throughputs.append(info.get('successful_transactions', 0))
        latencies.append(info.get('avg_latency', 0))
        malicious_nodes = htdcm.identify_malicious_nodes()
        malicious_detections.append(len(malicious_nodes))
        
        # In thông tin huấn luyện
        if episode % args.log_interval == 0:
            print(f"Episode {episode}/{args.episodes} - "
                 f"Reward: {episode_reward:.2f}, "
                 f"Avg Reward: {avg_reward:.2f}, "
                 f"Epsilon: {agent.epsilon:.4f}, "
                 f"Throughput: {transaction_throughputs[-1]}, "
                 f"Latency: {latencies[-1]:.2f}ms, "
                 f"Malicious Nodes: {len(malicious_nodes)}")
        
        # Lưu mô hình
        if episode % 100 == 0 or episode == args.episodes - 1:
            model_path = os.path.join(args.save_dir, f"dqn_model_ep{episode}.pth")
            agent.save(model_path)
            print(f"Đã lưu mô hình tại: {model_path}")
    
    # Lưu mô hình cuối cùng
    final_model_path = os.path.join(args.save_dir, "dqn_model_final.pth")
    agent.save(final_model_path)
    print(f"Đã lưu mô hình cuối cùng tại: {final_model_path}")
    
    # Lưu mô hình Federated Learning (nếu có)
    if fl_system is not None:
        fl_model_path = os.path.join(args.save_dir, "federated_model_final.pth")
        fl_system.save_global_model(fl_model_path)
        print(f"Đã lưu mô hình Federated Learning tại: {fl_model_path}")
    
    # Trả về dữ liệu hiệu suất
    return {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'transaction_throughputs': transaction_throughputs,
        'latencies': latencies,
        'malicious_detections': malicious_detections
    }

def evaluate_qtrust(env, agent, router, consensus, htdcm, args):
    """Đánh giá hiệu suất của hệ thống QTrust với mô hình đã huấn luyện."""
    print("Bắt đầu đánh giá hệ thống QTrust...")
    
    # Tải mô hình đã huấn luyện
    if args.model_path:
        agent.load(args.model_path)
        print(f"Đã tải mô hình từ: {args.model_path}")
    else:
        print("Không có đường dẫn mô hình để đánh giá. Sử dụng mô hình hiện tại.")
    
    # Số lượng episode để đánh giá
    eval_episodes = 50
    
    # Theo dõi hiệu suất
    rewards = []
    transaction_throughputs = []
    latencies = []
    energy_consumptions = []
    security_levels = []
    
    for episode in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_transactions = 0
        episode_latencies = []
        episode_energy = 0
        steps = 0
        
        while not done and steps < args.max_steps:
            # Sử dụng mô hình để chọn hành động tốt nhất (không có khám phá)
            action = agent.select_action(state, test_mode=True)
            
            # Định tuyến các giao dịch
            transaction_routes = router.find_optimal_paths_for_transactions(env.transaction_pool)
            
            # Thực hiện các giao dịch với giao thức đồng thuận thích ứng
            for tx in env.transaction_pool:
                # Chọn giao thức đồng thuận dựa trên giá trị giao dịch và tình trạng mạng
                trust_scores_dict = htdcm.get_node_trust_scores()
                protocol = consensus.select_protocol(
                    transaction_value=tx['value'],
                    network_congestion=env.get_congestion_level(),
                    trust_scores=list(trust_scores_dict.values())
                )
                
                result, latency, energy = consensus.execute_consensus(
                    tx, env.network, protocol
                )
                
                if result:
                    episode_transactions += 1
                
                episode_latencies.append(latency)
                episode_energy += energy
                
                # Cập nhật điểm tin cậy
                for node_id in tx.get('validator_nodes', []):
                    htdcm.update_node_trust(
                        node_id=node_id,
                        tx_success=result, 
                        response_time=latency, 
                        is_validator=True
                    )
            
            # Thực hiện bước trong môi trường
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Cập nhật trạng thái mạng cho router
            router.update_network_state(
                shard_congestion=env.get_congestion_data(),
                node_trust_scores=htdcm.get_node_trust_scores()
            )
        
        # Lưu kết quả
        rewards.append(episode_reward)
        transaction_throughputs.append(episode_transactions / steps if steps > 0 else 0)
        latencies.append(np.mean(episode_latencies) if episode_latencies else 0)
        energy_consumptions.append(episode_energy / steps if steps > 0 else 0)
        
        # Tính mức độ bảo mật dựa trên điểm tin cậy trung bình của mạng
        trust_scores = list(htdcm.get_node_trust_scores().values())
        security_level = np.mean(trust_scores) if trust_scores else 0
        security_levels.append(security_level)
        
        # In tiến trình
        print(f"Đánh giá Episode {episode+1}/{eval_episodes} - Reward: {episode_reward:.2f}, "
              f"Throughput: {transaction_throughputs[-1]:.4f}, Latency: {latencies[-1]:.2f}ms, "
              f"Energy: {energy_consumptions[-1]:.2f}, Security: {security_level:.4f}")
    
    # In kết quả tổng kết
    print("\n=== KẾT QUẢ ĐÁNH GIÁ QTrust ===")
    print(f"Phần thưởng trung bình: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"Throughput trung bình: {np.mean(transaction_throughputs):.4f} tx/step ± {np.std(transaction_throughputs):.4f}")
    print(f"Độ trễ trung bình: {np.mean(latencies):.4f}ms ± {np.std(latencies):.4f}")
    print(f"Tiêu thụ năng lượng trung bình: {np.mean(energy_consumptions):.4f} ± {np.std(energy_consumptions):.4f}")
    print(f"Mức độ bảo mật trung bình: {np.mean(security_levels):.4f} ± {np.std(security_levels):.4f}")
    
    # Trả về kết quả để có thể vẽ đồ thị hoặc phân tích thêm
    return {
        'rewards': rewards,
        'throughputs': transaction_throughputs,
        'latencies': latencies,
        'energy_consumptions': energy_consumptions,
        'security_levels': security_levels
    }

def plot_results(metrics, args, mode='train'):
    """Vẽ đồ thị kết quả."""
    print(f"Vẽ đồ thị kết quả {'huấn luyện' if mode == 'train' else 'đánh giá'}...")
    
    if mode == 'train':
        # Vẽ đồ thị phần thưởng
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['episode_rewards'], label='Episode Reward', alpha=0.6)
        plt.plot(metrics['avg_rewards'], label='Avg Reward (100 episodes)', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('QTrust Training Rewards')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'training_rewards.png'))
        
        # Vẽ đồ thị hiệu suất
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(metrics['transaction_throughputs'])
        plt.xlabel('Episode')
        plt.ylabel('Transactions')
        plt.title('Transaction Throughput')
        
        plt.subplot(3, 1, 2)
        plt.plot(metrics['latencies'])
        plt.xlabel('Episode')
        plt.ylabel('Latency (ms)')
        plt.title('Average Latency')
        
        plt.subplot(3, 1, 3)
        plt.plot(metrics['malicious_detections'])
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.title('Malicious Nodes Detected')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
    else:
        # Vẽ hộp đồ cho các chỉ số đánh giá
        plt.figure(figsize=(12, 10))
        
        metrics_data = [
            metrics['rewards'],
            metrics['throughputs'],
            metrics['latencies'],
            metrics['energy_consumptions'],
            metrics['security_levels']
        ]
        
        labels = [
            'Rewards',
            'Throughput (tx/step)',
            'Latency (ms)',
            'Energy Consumption',
            'Security Level'
        ]
        
        plt.boxplot(metrics_data, labels=labels)
        plt.title('QTrust Evaluation Metrics')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(args.save_dir, 'evaluation_metrics.png'))
    
    plt.close('all')

def main():
    """Hàm chính thực thi chương trình."""
    # Phân tích tham số
    args = parse_args()
    
    # Thiết lập môi trường và các thành phần
    env = setup_environment(args)
    router, consensus, htdcm = setup_components(env, args)
    agent = setup_dqn_agent(env, args)
    fl_system = setup_federated_learning(env, args, htdcm) if args.enable_federated else None
    
    # Huấn luyện hoặc đánh giá
    if args.eval:
        eval_metrics = evaluate_qtrust(env, agent, router, consensus, htdcm, args)
        plot_results(eval_metrics, args, mode='eval')
    else:
        train_metrics = train_qtrust(env, agent, router, consensus, htdcm, fl_system, args)
        plot_results(train_metrics, args, mode='train')
    
    print("Hoàn thành!")

if __name__ == "__main__":
    main() 