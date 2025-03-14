"""
Module mô phỏng chính cho hệ thống blockchain tối ưu hóa dựa trên DQN
"""

import time
import numpy as np
import random
import argparse
import os
import json
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

from dqn_blockchain_sim.blockchain.network import BlockchainNetwork
from dqn_blockchain_sim.agents.dqn_agent import MultiAgentDQNController
from dqn_blockchain_sim.federated_learning.federated_learner import FederatedLearner
from dqn_blockchain_sim.tdcm.trust_manager import TrustManager
from dqn_blockchain_sim.configs.simulation_config import (
    BLOCKCHAIN_CONFIG,
    DQN_CONFIG,
    FL_CONFIG,
    TDCM_CONFIG,
    SIMULATION_CONFIG
)


class Simulation:
    """
    Lớp mô phỏng chính, điều phối tất cả các thành phần
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Khởi tạo mô phỏng
        
        Args:
            config: Cấu hình mô phỏng, sử dụng mặc định nếu không cung cấp
        """
        self.config = config if config is not None else SIMULATION_CONFIG
        
        # Thiết lập seed cho tính nhất quán
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        
        # Khởi tạo mạng blockchain
        self.network = BlockchainNetwork(BLOCKCHAIN_CONFIG)
        
        # Khởi tạo trust manager
        self.trust_manager = TrustManager(TDCM_CONFIG)
        
        # Khởi tạo các thành phần DQN
        # Định nghĩa không gian trạng thái và hành động
        self.state_dim = 6  # Số chiều của vector trạng thái shard
        self.action_dim = 3  # Số hành động có thể (ví dụ: giữ nguyên, di chuyển nút, tái phân mảnh)
        self.global_state_dim = 8  # Số chiều của vector trạng thái toàn cục
        self.global_action_dim = 4  # Số hành động toàn cục
        
        # Khởi tạo bộ điều khiển DQN đa tác tử
        self.dqn_controller = MultiAgentDQNController(
            num_shards=BLOCKCHAIN_CONFIG["num_shards"],
            state_size=self.state_dim,
            action_size=self.action_dim,
            coordination_state_dim=self.global_state_dim,
            coordination_action_dim=self.global_action_dim,
            config=DQN_CONFIG
        )
        
        # Khởi tạo bộ học liên kết
        self.federated_learner = FederatedLearner(FL_CONFIG)
        
        # Khởi tạo các biến theo dõi mô phỏng
        self.current_step = 0
        self.current_time = 0.0
        self.stats_history = []
        self.attack_logs = []
        
        # Thêm nút vào trust manager
        self._initialize_trust_scores()
        
        # Thêm các shard như các khách hàng trong federated learning
        self._initialize_federated_clients()
        
    def _initialize_trust_scores(self) -> None:
        """
        Khởi tạo điểm tin cậy cho tất cả các nút trong mạng
        """
        for node_id, node_info in self.network.nodes.items():
            self.trust_manager.add_node(node_id, node_info)
    
    def _initialize_federated_clients(self) -> None:
        """
        Khởi tạo các khách hàng cho học liên kết
        """
        # Sử dụng các shard như các khách hàng trong mô hình học liên kết
        for shard_id, shard in self.network.shards.items():
            client_info = {
                'shard_id': shard_id,
                'node_count': len(shard.nodes),
                'reliability': 1.0
            }
            self.federated_learner.add_client(f"shard_{shard_id}", client_info)
            
        # Khởi tạo mô hình toàn cục với tham số từ một tác tử
        # (Trong triển khai thực tế, điều này có thể phức tạp hơn)
        initial_params = self.dqn_controller.get_all_model_parameters()
        if initial_params:
            first_agent_params = next(iter(initial_params.values()))
            self.federated_learner.initialize_global_model(first_agent_params)
    
    def run_simulation(self, num_steps: int = None) -> None:
        """
        Chạy mô phỏng
        
        Args:
            num_steps: Số bước mô phỏng, mặc định sử dụng giá trị từ cấu hình
        """
        if num_steps is None:
            # Tính số bước dựa trên thời gian mô phỏng từ cấu hình
            num_steps = self.config["duration"]
            
        # Chuẩn bị tqdm để hiển thị tiến độ
        progress_bar = tqdm(total=num_steps, desc="Mô phỏng")
        
        while self.current_step < num_steps:
            # Cập nhật thời gian
            self.current_time = self.current_step  # Giả định 1 bước = 1 giây
            
            # 1. Xử lý mạng blockchain
            network_stats = self.network.process_network(self.current_time)
            
            # 2. Tạo giao dịch mới
            self._generate_transactions()
            
            # 3. Kiểm tra và thực hiện các tác động DQN
            self._process_dqn_actions()
            
            # 4. Cập nhật điểm tin cậy
            self._update_trust_scores()
            
            # 5. Xử lý học liên kết
            self._process_federated_learning()
            
            # 6. Kiểm tra và mô phỏng tấn công nếu cần
            self._check_attack_scenarios()
            
            # 7. Thu thập thống kê
            self._collect_statistics()
            
            # 8. Tăng bước mô phỏng
            self.current_step += 1
            progress_bar.update(1)
            
        progress_bar.close()
        
        # Hiển thị kết quả nếu được yêu cầu
        if self.config["visualization"]:
            self.visualize_results()
            
        # Lưu kết quả nếu được yêu cầu
        if self.config["save_results"]:
            self.save_results()
    
    def _generate_transactions(self) -> None:
        """
        Tạo giao dịch mới cho mô phỏng
        """
        # Xác định số lượng giao dịch cần tạo
        tx_rate = BLOCKCHAIN_CONFIG["transaction_rate"]
        
        # Điều chỉnh tỷ lệ dựa trên phân phối
        if BLOCKCHAIN_CONFIG["transaction_distribution"] == "normal":
            # Mô phỏng dao động trong ngày với phân phối chuẩn
            hour_of_day = (self.current_time / 3600) % 24
            rate_factor = 1.0 + 0.5 * np.sin(hour_of_day / 24.0 * 2 * np.pi)
            tx_rate = int(tx_rate * rate_factor)
        elif BLOCKCHAIN_CONFIG["transaction_distribution"] == "poisson":
            # Phân phối Poisson
            tx_rate = np.random.poisson(tx_rate)
            
        # Tạo giao dịch mới
        num_transactions = max(1, int(tx_rate / 60))  # Mỗi bước mô phỏng là 1 giây
        
        for _ in range(num_transactions):
            tx = self.network.generate_random_transaction()
            self.network.add_transaction(tx)
    
    def _process_dqn_actions(self) -> None:
        """
        Xử lý các hành động từ tác tử DQN
        """
        # Trong quá trình khởi động, chỉ thu thập dữ liệu
        if self.current_time < self.config["warmup_time"]:
            return
            
        # Trích xuất các trạng thái từ hệ thống
        network_state = self.network.get_network_stats()
        shard_states = {
            shard_id: shard.get_stats() 
            for shard_id, shard in self.network.shards.items()
        }
        
        # Xử lý hành động cho mỗi shard
        for shard_id, shard_state in shard_states.items():
            # Trích xuất trạng thái cũ (nếu có)
            old_state = None
            if len(self.stats_history) > 0:
                last_stats = self.stats_history[-1]
                if 'shard_stats' in last_stats and shard_id in last_stats['shard_stats']:
                    old_state = last_stats['shard_stats'][shard_id]
            
            if old_state is None:
                # Nếu không có trạng thái cũ, bỏ qua chu kỳ này
                continue
                
            # Biểu diễn vector trạng thái cho DQN
            state_vector = self.dqn_controller.shard_agents[shard_id].get_state_representation(shard_state)
            
            # Chọn hành động
            action = self.dqn_controller.shard_agents[shard_id].select_action(state_vector)
            
            # Thực hiện hành động
            reward = self._execute_shard_action(shard_id, action)
            
            # Lấy trạng thái mới sau khi thực hiện hành động
            new_shard_state = self.network.shards[shard_id].get_stats()
            new_state_vector = self.dqn_controller.shard_agents[shard_id].get_state_representation(new_shard_state)
            
            # Lưu trải nghiệm vào bộ nhớ
            done = False  # Trong mô phỏng liên tục, không có trạng thái "kết thúc"
            self.dqn_controller.shard_agents[shard_id].add_experience(
                state_vector, action, reward, new_state_vector, done
            )
            
        # Xử lý hành động phối hợp toàn cục
        if self.dqn_controller.has_coordinator:
            # Tạo vector trạng thái toàn cục
            global_state = self.dqn_controller.get_global_state(network_state)
            
            # Lấy trạng thái mạng cũ
            old_network_state = None
            if len(self.stats_history) > 0:
                old_network_state = self.stats_history[-1]
            
            if old_network_state is not None:
                # Chọn hành động toàn cục
                global_action = self.dqn_controller.select_coordination_action(global_state)
                
                # Thực hiện hành động toàn cục
                global_reward = self._execute_global_action(global_action)
                
                # Lấy trạng thái mới
                new_network_state = self.network.get_network_stats()
                new_global_state = self.dqn_controller.get_global_state(new_network_state)
                
                # Lưu trải nghiệm
                done = False
                self.dqn_controller.coordination_agent.add_experience(
                    global_state, global_action, global_reward, new_global_state, done
                )
                
        # Huấn luyện tác tử DQN
        if self.current_step % 10 == 0:  # Huấn luyện sau mỗi 10 bước
            self.dqn_controller.train_agents(DQN_CONFIG["batch_size"])
    
    def _execute_shard_action(self, shard_id: int, action: int) -> float:
        """
        Thực hiện hành động cho một shard cụ thể
        
        Args:
            shard_id: ID của shard
            action: Hành động được chọn
            
        Returns:
            Phần thưởng nhận được
        """
        reward = 0.0
        shard = self.network.shards[shard_id]
        
        if action == 0:
            # Hành động 0: Không làm gì (theo dõi)
            # Không có thay đổi, phần thưởng nhỏ cho việc tiết kiệm tài nguyên
            reward = 0.1
            
        elif action == 1:
            # Hành động 1: Di chuyển một nút nếu shard quá tải
            if shard.congestion_level > 0.7 and len(shard.nodes) > shard.min_nodes:
                # Tìm shard ít tải nhất
                min_congestion = float('inf')
                target_shard_id = None
                
                for sid, s in self.network.shards.items():
                    if sid != shard_id and s.congestion_level < min_congestion:
                        min_congestion = s.congestion_level
                        target_shard_id = sid
                
                if target_shard_id is not None and min_congestion < shard.congestion_level:
                    # Chọn một nút để di chuyển
                    node_to_move = random.choice(list(shard.nodes))
                    if self.network.move_node(node_to_move, target_shard_id):
                        # Di chuyển thành công
                        congestion_diff = shard.congestion_level - min_congestion
                        reward = congestion_diff * 5.0  # Phần thưởng tỷ lệ thuận với mức giảm tắc nghẽn
                    else:
                        # Di chuyển thất bại
                        reward = -0.5
                else:
                    # Không có shard đích phù hợp
                    reward = -0.2
            else:
                # Shard không quá tải hoặc không đủ nút để di chuyển
                reward = -0.3
                
        elif action == 2:
            # Hành động 2: Đề xuất tái phân mảnh nếu cần
            if shard.congestion_level > BLOCKCHAIN_CONFIG["reshard_threshold"]:
                # Đề xuất tái phân mảnh
                changes = self.network.optimize_shards()
                if changes:
                    # Tái phân mảnh thành công
                    reward = 2.0 + len(changes) * 0.5  # Phần thưởng dựa trên số nút được di chuyển
                else:
                    # Không có thay đổi
                    reward = -0.2
            else:
                # Tái phân mảnh không cần thiết
                reward = -1.0
                
        return reward
    
    def _execute_global_action(self, action: int) -> float:
        """
        Thực hiện hành động toàn cục
        
        Args:
            action: Hành động được chọn
            
        Returns:
            Phần thưởng nhận được
        """
        reward = 0.0
        
        if action == 0:
            # Hành động 0: Theo dõi (không làm gì)
            reward = 0.1
            
        elif action == 1:
            # Hành động 1: Cân bằng tải toàn bộ mạng
            changes = self.network.optimize_shards()
            if changes:
                reward = 1.0 + len(changes) * 0.3
            else:
                reward = -0.5
                
        elif action == 2:
            # Hành động 2: Tái phân bổ nút dựa trên điểm tin cậy
            # Xác định các nút có điểm tin cậy thấp
            trust_scores = self.trust_manager.get_all_trust_scores()
            low_trust_nodes = [node_id for node_id, score in trust_scores.items() if score < 0.6]
            
            if low_trust_nodes:
                # Di chuyển các nút tin cậy thấp vào các shard ít quan trọng
                moved_count = 0
                low_importance_shards = sorted(
                    self.network.shards.items(), 
                    key=lambda x: x[1].total_transactions
                )[:2]  # 2 shard có ít giao dịch nhất
                
                if low_importance_shards:
                    low_imp_shard_ids = [sid for sid, _ in low_importance_shards]
                    for node_id in low_trust_nodes[:5]:  # Giới hạn 5 nút một lần
                        current_shard = self.network.get_node_shard(node_id)
                        if current_shard is not None and current_shard not in low_imp_shard_ids:
                            target_shard = random.choice(low_imp_shard_ids)
                            if self.network.move_node(node_id, target_shard):
                                moved_count += 1
                                
                if moved_count > 0:
                    reward = moved_count * 0.5
                else:
                    reward = -0.3
            else:
                reward = 0.2  # Tốt khi không có nút tin cậy thấp
                
        elif action == 3:
            # Hành động 3: Tối ưu hóa giao dịch xuyên mảnh
            # Đánh giá mô hình kết nối shard
            if hasattr(self.network, 'shard_graph'):
                congested_pairs = []
                
                # Tìm các cặp shard có nhiều giao dịch xuyên mảnh
                for shard_id, shard in self.network.shards.items():
                    cross_txs = shard.cross_shard_transactions
                    if cross_txs > 0:
                        for tx in shard.cross_shard_queue:
                            if tx.target_shard is not None:
                                pair = (shard_id, tx.target_shard)
                                congested_pairs.append(pair)
                
                # Đếm số lần xuất hiện của mỗi cặp
                pair_counts = {}
                for pair in congested_pairs:
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                
                # Tìm các cặp shard thường xuyên giao tiếp nhưng không kết nối trực tiếp
                improvements = 0
                for (src, dst), count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True):
                    if count > 5 and not self.network.shard_graph.has_edge(src, dst):
                        # Thêm kết nối trực tiếp với độ trễ thấp
                        self.network.shard_graph.add_edge(src, dst, latency=50)
                        improvements += 1
                        if improvements >= 2:  # Giới hạn 2 cải tiến mỗi lần
                            break
                
                if improvements > 0:
                    reward = improvements * 1.5
                else:
                    reward = -0.1
            else:
                reward = -0.2
                
        return reward
    
    def _update_trust_scores(self) -> None:
        """
        Cập nhật điểm tin cậy cho tất cả các nút
        """
        # Cập nhật điểm tin cậy theo tần suất được cấu hình
        if self.current_step % int(TDCM_CONFIG["trust_update_frequency"]) == 0:
            # Thu thập chỉ số từ tất cả các nút
            metrics_batch = {}
            
            for node_id, node_info in self.network.nodes.items():
                # Lấy thông tin hiệu suất
                shard_id = self.network.get_node_shard(node_id)
                if shard_id is not None:
                    shard = self.network.shards[shard_id]
                    
                    # Tạo chỉ số cho đánh giá tin cậy
                    metrics = {
                        "uptime": node_info.get("uptime", 1.0),
                        "performance": node_info.get("compute_power", 0.5),
                        "security_incidents": 0,  # Mặc định không có sự cố
                        "energy_efficiency": node_info.get("energy_efficiency", 0.7)
                    }
                    
                    metrics_batch[node_id] = metrics
            
            # Cập nhật hàng loạt
            self.trust_manager.batch_update_trust_scores(metrics_batch)
    
    def _process_federated_learning(self) -> None:
        """
        Xử lý chu trình học liên kết (federated learning)
        """
        # Bỏ qua trong giai đoạn khởi động
        if self.current_time < self.config["warmup_time"]:
            return
            
        # Thực hiện các bước học liên kết
        round_num, selected_clients = self.federated_learner.step()
        
        # Xử lý khách hàng được chọn (các shard)
        for client_id in selected_clients:
            # Phân tích ID khách hàng để lấy shard_id
            parts = client_id.split('_')
            if len(parts) == 2 and parts[0] == 'shard':
                try:
                    shard_id = int(parts[1])
                    if shard_id in self.dqn_controller.shard_agents:
                        # Lấy mô hình toàn cục hiện tại
                        global_model = self.federated_learner.get_model_for_client(client_id)
                        
                        if global_model is not None:
                            # Xác định xem có nên cập nhật mô hình cục bộ không
                            # (Trong triển khai thực, cần nhiều logic hơn)
                            if random.random() < 0.8:  # 80% xác suất tham gia
                                # Lấy tham số cục bộ để gửi về máy chủ
                                local_params = self.dqn_controller.shard_agents[shard_id].get_model_parameters()
                                
                                # Chuẩn bị metrics
                                metrics = {
                                    'loss': random.uniform(0.1, 2.0),  # Giả lập giá trị loss
                                    'gradient_norm': random.uniform(0.5, 5.0)  # Giả lập norm gradient
                                }
                                
                                # Gửi cập nhật
                                self.federated_learner.receive_client_update(
                                    client_id, local_params, metrics
                                )
                except ValueError:
                    pass
                    
        # Kiểm tra xem có nên áp dụng mô hình toàn cục mới không
        # Thực hiện sau mỗi kỳ tổng hợp (thường là sau vài vòng FL)
        if round_num > 0 and round_num % FL_CONFIG["aggregation_frequency"] == 0:
            # Lấy mô hình toàn cục đã được tổng hợp
            global_model = self.federated_learner.get_global_model()
            
            if global_model is not None:
                # Áp dụng mô hình toàn cục cho tất cả các tác tử
                self.dqn_controller.set_model_parameters({
                    f"shard_{i}": global_model 
                    for i in range(BLOCKCHAIN_CONFIG["num_shards"])
                })
    
    def _check_attack_scenarios(self) -> None:
        """
        Kiểm tra và mô phỏng các kịch bản tấn công
        """
        # Kiểm tra từng kịch bản tấn công trong cấu hình
        for attack in self.config["attack_scenarios"]:
            # Kiểm tra xem có phải thời điểm để bắt đầu tấn công không
            if (attack["start_time"] <= self.current_time < 
                attack["start_time"] + attack["duration"]):
                
                # Mô phỏng tấn công
                attack_result = self.network.simulate_attack(
                    attack["type"], 
                    attack["intensity"]
                )
                
                # Ghi lại thông tin tấn công
                attack_log = {
                    'step': self.current_step,
                    'time': self.current_time,
                    'attack_type': attack["type"],
                    'intensity': attack["intensity"],
                    'affected_shards': attack_result['affected_shards'],
                    'affected_nodes': attack_result['affected_nodes']
                }
                
                self.attack_logs.append(attack_log)
                
                # Cập nhật điểm tin cậy cho các nút bị ảnh hưởng
                for node_id in attack_result['affected_nodes']:
                    if node_id in self.trust_manager.nodes:
                        # Giảm điểm tin cậy của nút bị ảnh hưởng
                        metrics = {
                            "security_incidents": random.randint(1, 5)
                        }
                        self.trust_manager.update_node_metrics(node_id, metrics)
    
    def _collect_statistics(self) -> None:
        """
        Thu thập thống kê từ tất cả các thành phần của hệ thống
        """
        # Thu thập thống kê từ mạng blockchain
        network_stats = self.network.get_network_stats()
        
        # Thu thập thông tin về điểm tin cậy
        trust_stats = self.trust_manager.get_statistics()
        
        # Thu thập thông tin về học liên kết
        fl_stats = self.federated_learner.get_statistics()
        
        # Kết hợp tất cả thống kê
        stats = {
            'step': self.current_step,
            'time': self.current_time,
            'network': network_stats,
            'trust': trust_stats,
            'federated_learning': fl_stats,
            'dqn': {
                'global_steps': self.dqn_controller.global_steps,
                'avg_loss': np.mean(self.dqn_controller.global_losses[-100:]) 
                    if self.dqn_controller.global_losses else 0,
                'epsilon': self.dqn_controller.shard_agents[0].epsilon 
                    if 0 in self.dqn_controller.shard_agents else 0
            }
        }
        
        self.stats_history.append(stats)
    
    def visualize_results(self) -> None:
        """
        Hiển thị kết quả mô phỏng bằng đồ thị
        """
        if not self.stats_history:
            print("Không có dữ liệu thống kê để hiển thị!")
            return
            
        # Chuẩn bị dữ liệu
        steps = [s['step'] for s in self.stats_history]
        congestion = [s['network']['network_congestion'] for s in self.stats_history]
        latency = [s['network']['avg_network_latency'] for s in self.stats_history]
        energy = [s['network']['total_energy_consumption'] for s in self.stats_history]
        trust_scores = [s['trust']['avg_trust_score'] for s in self.stats_history]
        
        # Tạo hình
        plt.figure(figsize=(15, 10))
        
        # Đồ thị tắc nghẽn mạng
        plt.subplot(2, 2, 1)
        plt.plot(steps, congestion, 'b-')
        plt.title('Tắc nghẽn mạng')
        plt.xlabel('Bước')
        plt.ylabel('Mức độ tắc nghẽn')
        plt.grid(True)
        
        # Đồ thị độ trễ
        plt.subplot(2, 2, 2)
        plt.plot(steps, latency, 'r-')
        plt.title('Độ trễ mạng trung bình')
        plt.xlabel('Bước')
        plt.ylabel('Độ trễ (ms)')
        plt.grid(True)
        
        # Đồ thị tiêu thụ năng lượng
        plt.subplot(2, 2, 3)
        plt.plot(steps, energy, 'g-')
        plt.title('Tiêu thụ năng lượng')
        plt.xlabel('Bước')
        plt.ylabel('Năng lượng (kWh)')
        plt.grid(True)
        
        # Đồ thị điểm tin cậy
        plt.subplot(2, 2, 4)
        plt.plot(steps, trust_scores, 'm-')
        plt.title('Điểm tin cậy trung bình')
        plt.xlabel('Bước')
        plt.ylabel('Điểm tin cậy')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Lưu hình nếu cần
        if self.config["save_results"]:
            plt.savefig("simulation_results.png")
            
        plt.show()
        
        # Hiển thị thêm thông tin về các cuộc tấn công
        if self.attack_logs:
            print("\nThông tin về các cuộc tấn công:")
            for attack in self.attack_logs:
                print(f"Bước {attack['step']}: Tấn công {attack['attack_type']} " +
                      f"với cường độ {attack['intensity']}, ảnh hưởng đến " +
                      f"{len(attack['affected_nodes'])} nút và " +
                      f"{len(attack['affected_shards'])} shard.")
    
    def _serialize_numpy_objects(self, obj):
        """
        Hàm tiện ích để chuyển đổi các đối tượng numpy thành các đối tượng có thể chuyển đổi JSON
        
        Args:
            obj: Đối tượng cần xử lý
            
        Returns:
            Đối tượng đã được xử lý, có thể chuyển đổi JSON
        """
        if isinstance(obj, dict):
            return {k: self._serialize_numpy_objects(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy_objects(v) for v in obj]
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def save_results(self) -> None:
        """
        Lưu kết quả mô phỏng
        """
        # Tạo thư mục results nếu chưa tồn tại
        os.makedirs("results", exist_ok=True)
        
        # Lưu thống kê
        with open("results/stats_history.json", "w") as f:
            # Chuyển đổi tất cả các đối tượng numpy
            serializable_stats = [self._serialize_numpy_objects(stat) for stat in self.stats_history]
            json.dump(serializable_stats, f, indent=2)
            
        # Lưu nhật ký tấn công
        with open("results/attack_logs.json", "w") as f:
            # Chuyển đổi tất cả các đối tượng numpy
            serializable_logs = self._serialize_numpy_objects(self.attack_logs)
            json.dump(serializable_logs, f, indent=2)
            
        # Lưu mô hình DQN
        os.makedirs("results/models", exist_ok=True)
        self.dqn_controller.save_all_agents("results/models")
        
        print(f"Đã lưu kết quả vào thư mục 'results/'")


def main():
    """
    Hàm chính để chạy mô phỏng từ dòng lệnh
    """
    parser = argparse.ArgumentParser(description='Mô phỏng blockchain tối ưu hóa bằng DQN')
    parser.add_argument('--steps', type=int, default=None, 
                      help='Số bước mô phỏng (mặc định: giá trị từ cấu hình)')
    parser.add_argument('--visualize', action='store_true', 
                      help='Hiển thị kết quả mô phỏng')
    parser.add_argument('--save', action='store_true', 
                      help='Lưu kết quả mô phỏng')
    
    args = parser.parse_args()
    
    # Cập nhật cấu hình
    config = SIMULATION_CONFIG.copy()
    if args.visualize:
        config["visualization"] = True
    if args.save:
        config["save_results"] = True
        
    # Tạo và chạy mô phỏng
    simulation = Simulation(config)
    simulation.run_simulation(args.steps)


if __name__ == "__main__":
    main() 