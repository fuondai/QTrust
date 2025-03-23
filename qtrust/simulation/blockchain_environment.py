import gym
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
from gym import spaces

class BlockchainEnvironment(gym.Env):
    """
    Môi trường mô phỏng blockchain với sharding cho Deep Reinforcement Learning.
    Môi trường này mô phỏng một mạng blockchain với nhiều shard và giao dịch xuyên shard.
    """
    
    def __init__(self, 
                 num_shards: int = 4, 
                 num_nodes_per_shard: int = 10,
                 max_transactions_per_step: int = 100,
                 transaction_value_range: Tuple[float, float] = (0.1, 100.0),
                 max_steps: int = 1000,
                 latency_penalty: float = 0.5,
                 energy_penalty: float = 0.3,
                 throughput_reward: float = 1.0,
                 security_reward: float = 0.8):
        """
        Khởi tạo môi trường blockchain với sharding.
        
        Args:
            num_shards: Số lượng shard trong mạng
            num_nodes_per_shard: Số lượng node trong mỗi shard
            max_transactions_per_step: Số lượng giao dịch tối đa mỗi bước
            transaction_value_range: Phạm vi giá trị giao dịch (min, max)
            max_steps: Số bước tối đa cho mỗi episode
            latency_penalty: Hệ số phạt cho độ trễ
            energy_penalty: Hệ số phạt cho tiêu thụ năng lượng
            throughput_reward: Hệ số thưởng cho throughput
            security_reward: Hệ số thưởng cho bảo mật
        """
        super(BlockchainEnvironment, self).__init__()
        
        self.num_shards = num_shards
        self.num_nodes_per_shard = num_nodes_per_shard
        self.max_transactions_per_step = max_transactions_per_step
        self.transaction_value_range = transaction_value_range
        self.max_steps = max_steps
        
        # Các hệ số reward/penalty
        self.latency_penalty = latency_penalty
        self.energy_penalty = energy_penalty
        self.throughput_reward = throughput_reward
        self.security_reward = security_reward
        
        # Số bước hiện tại
        self.current_step = 0
        
        # Khởi tạo không gian trạng thái và hành động
        self._init_state_action_space()
        
        # Khởi tạo mạng blockchain
        self._init_blockchain_network()
        
        # Khởi tạo transaction pool
        self.transaction_pool = []
        
        # Thống kê hiệu suất
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'security_score': []
        }
    
    def _init_state_action_space(self):
        """Khởi tạo không gian trạng thái và hành động."""
        # Không gian trạng thái:
        # - Mức độ tắc nghẽn mạng cho mỗi shard (0.0-1.0)
        # - Giá trị giao dịch trung bình trong mỗi shard
        # - Điểm tin cậy trung bình của các node trong mỗi shard (0.0-1.0)
        # - Tỷ lệ giao dịch thành công gần đây
        
        # Mỗi shard có 4 đặc trưng, cộng với 4 đặc trưng toàn cục
        num_features = self.num_shards * 4 + 4
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=float('inf'), 
            shape=(num_features,), 
            dtype=np.float32
        )
        
        # Không gian hành động:
        # - Lựa chọn shard đích cho một giao dịch (0 to num_shards-1)
        # - Lựa chọn giao thức đồng thuận (0: Fast BFT, 1: PBFT, 2: Robust BFT)
        self.action_space = spaces.MultiDiscrete([self.num_shards, 3])
        
        # Định nghĩa không gian trạng thái và hành động cho một cách nhìn dễ hiểu hơn
        self.state_space = {
            'network_congestion': [0.0, 1.0],  # Mức độ tắc nghẽn
            'transaction_value': [self.transaction_value_range[0], self.transaction_value_range[1]],
            'trust_scores': [0.0, 1.0],  # Điểm tin cậy
            'success_rate': [0.0, 1.0]   # Tỷ lệ thành công
        }
        
        self.action_space_dict = {
            'routing_decision': list(range(self.num_shards)),
            'consensus_selection': ['Fast_BFT', 'PBFT', 'Robust_BFT']
        }
    
    def _init_blockchain_network(self):
        """Khởi tạo mạng blockchain với shards và nodes."""
        # Sử dụng networkx để biểu diễn mạng blockchain
        self.network = nx.Graph()
        
        # Khởi tạo các node cho mỗi shard
        self.shards = []
        total_nodes = 0
        
        for shard_id in range(self.num_shards):
            shard_nodes = []
            for i in range(self.num_nodes_per_shard):
                node_id = total_nodes + i
                # Thêm node vào mạng
                self.network.add_node(
                    node_id, 
                    shard_id=shard_id,
                    trust_score=np.random.uniform(0.5, 1.0),  # Điểm tin cậy ban đầu
                    processing_power=np.random.uniform(0.7, 1.0),  # Khả năng xử lý (0.7-1.0)
                    energy_efficiency=np.random.uniform(0.6, 0.9)  # Hiệu suất năng lượng (0.6-0.9)
                )
                shard_nodes.append(node_id)
            
            self.shards.append(shard_nodes)
            total_nodes += self.num_nodes_per_shard
        
        # Tạo kết nối giữa các node trong cùng một shard (đầy đủ kết nối)
        for shard_nodes in self.shards:
            for i in range(len(shard_nodes)):
                for j in range(i + 1, len(shard_nodes)):
                    # Độ trễ từ 1ms đến 10ms cho các node trong cùng shard
                    self.network.add_edge(
                        shard_nodes[i], 
                        shard_nodes[j], 
                        latency=np.random.uniform(1, 10),
                        bandwidth=np.random.uniform(50, 100)  # Mbps
                    )
        
        # Tạo kết nối giữa các shard (một số kết nối ngẫu nhiên)
        for i in range(self.num_shards):
            for j in range(i + 1, self.num_shards):
                # Chọn ngẫu nhiên 3 node từ mỗi shard để kết nối
                nodes_from_shard_i = np.random.choice(self.shards[i], 3, replace=False)
                nodes_from_shard_j = np.random.choice(self.shards[j], 3, replace=False)
                
                for node_i in nodes_from_shard_i:
                    for node_j in nodes_from_shard_j:
                        # Độ trễ từ 10ms đến 50ms cho các node giữa các shard
                        self.network.add_edge(
                            node_i, 
                            node_j, 
                            latency=np.random.uniform(10, 50),
                            bandwidth=np.random.uniform(10, 50)  # Mbps
                        )
        
        # Thiết lập trạng thái congestion ban đầu cho mỗi shard
        self.shard_congestion = np.random.uniform(0.1, 0.3, self.num_shards)
        
        # Thiết lập trạng thái hiện tại cho consensus protocol của mỗi shard
        # 0: Fast BFT, 1: PBFT, 2: Robust BFT
        self.shard_consensus = np.zeros(self.num_shards, dtype=np.int32)
    
    def _generate_transactions(self) -> List[Dict[str, Any]]:
        """Tạo ngẫu nhiên các giao dịch cho bước hiện tại."""
        num_transactions = np.random.randint(1, self.max_transactions_per_step + 1)
        transactions = []
        
        for _ in range(num_transactions):
            # Chọn ngẫu nhiên shard nguồn và đích
            source_shard = np.random.randint(0, self.num_shards)
            # 70% khả năng giao dịch nội bộ trong shard, 30% là cross-shard
            if np.random.random() < 0.7:
                destination_shard = source_shard
            else:
                possible_destinations = [s for s in range(self.num_shards) if s != source_shard]
                destination_shard = np.random.choice(possible_destinations)
            
            # Chọn ngẫu nhiên node nguồn và đích
            source_node = np.random.choice(self.shards[source_shard])
            destination_node = np.random.choice(self.shards[destination_shard])
            
            # Tạo giá trị ngẫu nhiên cho giao dịch
            value = np.random.uniform(
                self.transaction_value_range[0], 
                self.transaction_value_range[1]
            )
            
            # Độ ưu tiên của giao dịch (0.0-1.0)
            priority = np.random.uniform(0.0, 1.0)
            
            transactions.append({
                'id': f"tx_{self.current_step}_{len(transactions)}",
                'source_shard': source_shard,
                'destination_shard': destination_shard,
                'source_node': source_node,
                'destination_node': destination_node,
                'value': value,
                'priority': priority,
                'size': np.random.uniform(1, 5),  # kB
                'created_at': self.current_step,
                'status': 'pending',
                'routed_path': [],
                'consensus_protocol': None,
                'completion_time': None,
                'energy_consumed': 0.0
            })
        
        return transactions
    
    def get_state(self) -> np.ndarray:
        """Lấy trạng thái hiện tại của môi trường."""
        state = []
        
        # Thông tin cho mỗi shard
        for shard_id in range(self.num_shards):
            shard_nodes = self.shards[shard_id]
            
            # Mức độ tắc nghẽn
            state.append(self.shard_congestion[shard_id])
            
            # Giá trị giao dịch trung bình trong shard
            shard_txs = [tx for tx in self.transaction_pool 
                         if tx['destination_shard'] == shard_id and tx['status'] == 'pending']
            if shard_txs:
                avg_value = np.mean([tx['value'] for tx in shard_txs])
            else:
                avg_value = 0.0
            state.append(avg_value)
            
            # Điểm tin cậy trung bình
            avg_trust = np.mean([self.network.nodes[node]['trust_score'] for node in shard_nodes])
            state.append(avg_trust)
            
            # Tỷ lệ giao dịch thành công gần đây
            recent_txs = [tx for tx in self.transaction_pool 
                          if tx['destination_shard'] == shard_id and tx['status'] != 'pending']
            if recent_txs:
                success_rate = len([tx for tx in recent_txs if tx['status'] == 'completed']) / len(recent_txs)
            else:
                success_rate = 1.0
            state.append(success_rate)
        
        # Thông tin toàn cục
        # Tổng số giao dịch đang chờ xử lý
        pending_txs = len([tx for tx in self.transaction_pool if tx['status'] == 'pending'])
        state.append(pending_txs / max(1, self.max_transactions_per_step))
        
        # Độ trễ trung bình của mạng
        edge_latencies = [data['latency'] for _, _, data in self.network.edges(data=True)]
        avg_latency = np.mean(edge_latencies) if edge_latencies else 0
        normalized_latency = min(1.0, avg_latency / 50.0)  # Chuẩn hóa với giá trị tối đa 50ms
        state.append(normalized_latency)
        
        # Tỷ lệ giao dịch xuyên shard
        all_txs = len(self.transaction_pool)
        cross_shard_txs = len([tx for tx in self.transaction_pool 
                              if tx['source_shard'] != tx['destination_shard']])
        cross_shard_ratio = cross_shard_txs / max(1, all_txs)
        state.append(cross_shard_ratio)
        
        # Tỷ lệ đồng thuận hiện tại (% của mỗi loại)
        consensus_counts = np.bincount(self.shard_consensus, minlength=3)
        consensus_ratio = consensus_counts / self.num_shards
        state.extend(consensus_ratio)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self, action, transaction) -> float:
        """
        Tính toán phần thưởng cho một hành động dựa trên các metrics và kết quả giao dịch.
        
        Args:
            action: Hành động được chọn (shard đích, giao thức đồng thuận)
            transaction: Giao dịch hiện tại
        
        Returns:
            float: Giá trị phần thưởng
        """
        destination_shard = action[0]
        consensus_protocol = action[1]
        
        # Kiểm tra nếu giao dịch đã hoàn thành
        if transaction['status'] != 'completed':
            return -1.0  # Phạt nếu giao dịch không thành công
        
        # Tính độ trễ (thời gian hoàn thành - thời gian tạo)
        latency = transaction['completion_time'] - transaction['created_at']
        normalized_latency = min(1.0, latency / 10.0)  # Chuẩn hóa với thời gian tối đa 10 bước
        
        # Tính tiêu thụ năng lượng
        energy_consumed = transaction['energy_consumed']
        normalized_energy = min(1.0, energy_consumed / 100.0)  # Chuẩn hóa với giá trị tối đa 100
        
        # Đánh giá bảo mật dựa trên giao thức đồng thuận và giá trị giao dịch
        security_score = 0.0
        if consensus_protocol == 0:  # Fast BFT
            # Phù hợp với giao dịch giá trị thấp và tắc nghẽn thấp
            if transaction['value'] < 10.0 and self.shard_congestion[destination_shard] < 0.5:
                security_score = 0.7
            else:
                security_score = 0.3
        elif consensus_protocol == 1:  # PBFT
            # Phù hợp với hầu hết các giao dịch
            security_score = 0.8
        else:  # Robust BFT
            # Phù hợp với giao dịch giá trị cao
            if transaction['value'] > 50.0:
                security_score = 1.0
            else:
                security_score = 0.6
        
        # Tính phần thưởng tổng hợp
        throughput_reward = self.throughput_reward * 1.0  # Giao dịch thành công
        latency_penalty = self.latency_penalty * normalized_latency
        energy_penalty = self.energy_penalty * normalized_energy
        security_reward = self.security_reward * security_score
        
        # Cộng tất cả các thành phần
        reward = throughput_reward - latency_penalty - energy_penalty + security_reward
        
        return reward
    
    def _process_transaction(self, transaction, action) -> Tuple[Dict[str, Any], float]:
        """
        Xử lý một giao dịch dựa trên hành động đã chọn.
        
        Args:
            transaction: Giao dịch cần xử lý
            action: Hành động được chọn (shard đích, giao thức đồng thuận)
        
        Returns:
            Tuple[Dict, float]: Giao dịch sau khi xử lý và phần thưởng
        """
        destination_shard = action[0]
        consensus_protocol = action[1]
        
        # Cập nhật đường dẫn định tuyến
        transaction['routed_path'].append(destination_shard)
        transaction['consensus_protocol'] = self.action_space_dict['consensus_selection'][consensus_protocol]
        
        # Tính toán độ trễ dựa trên tình trạng tắc nghẽn và giao thức đồng thuận
        base_latency = 1.0
        
        # Tính thêm độ trễ nếu là giao dịch xuyên shard
        if transaction['source_shard'] != destination_shard:
            # Tìm độ trễ trung bình giữa hai shard
            cross_shard_edges = [
                (u, v) for u, v, data in self.network.edges(data=True)
                if self.network.nodes[u]['shard_id'] == transaction['source_shard'] and
                self.network.nodes[v]['shard_id'] == destination_shard
            ]
            
            if cross_shard_edges:
                # Tính độ trễ trung bình của các kết nối giữa hai shard
                avg_latency = np.mean([
                    self.network.edges[u, v]['latency'] 
                    for u, v in cross_shard_edges
                ])
                base_latency += avg_latency / 10.0  # Chuẩn hóa theo ms
        
        # Thêm độ trễ dựa trên tình trạng tắc nghẽn
        congestion_latency = self.shard_congestion[destination_shard] * 3.0
        
        # Thêm độ trễ dựa trên giao thức đồng thuận
        consensus_latency = 0.0
        if consensus_protocol == 0:  # Fast BFT
            consensus_latency = 0.5
        elif consensus_protocol == 1:  # PBFT
            consensus_latency = 1.0
        else:  # Robust BFT
            consensus_latency = 2.0
        
        # Tổng độ trễ
        total_latency = base_latency + congestion_latency + consensus_latency
        
        # Tính toán tiêu thụ năng lượng
        base_energy = 10.0  # Năng lượng cơ sở cho một giao dịch
        
        # Thêm năng lượng cho cross-shard
        if transaction['source_shard'] != destination_shard:
            base_energy += 15.0
        
        # Thêm năng lượng dựa trên giao thức đồng thuận
        consensus_energy = 0.0
        if consensus_protocol == 0:  # Fast BFT
            consensus_energy = 5.0
        elif consensus_protocol == 1:  # PBFT
            consensus_energy = 15.0
        else:  # Robust BFT
            consensus_energy = 30.0
        
        # Tổng năng lượng tiêu thụ
        total_energy = base_energy + consensus_energy
        
        # Cập nhật thông tin giao dịch
        transaction['energy_consumed'] = total_energy
        
        # Xác định xem giao dịch có thành công hay không
        success_prob = 0.95  # Tỷ lệ thành công mặc định
        
        # Giảm tỷ lệ thành công nếu tắc nghẽn cao
        if self.shard_congestion[destination_shard] > 0.8:
            success_prob -= 0.2
        
        # Tăng tỷ lệ thành công với giao thức đồng thuận mạnh hơn
        if consensus_protocol == 2:  # Robust BFT
            success_prob += 0.05
        
        # Quyết định kết quả giao dịch
        if np.random.random() < success_prob:
            transaction['status'] = 'completed'
            transaction['completion_time'] = self.current_step + total_latency
        else:
            transaction['status'] = 'failed'
        
        # Cập nhật mức độ tắc nghẽn của shard đích
        self.shard_congestion[destination_shard] = min(
            1.0, 
            self.shard_congestion[destination_shard] + 0.05
        )
        
        # Tính toán phần thưởng
        reward = self._calculate_reward(action, transaction)
        
        return transaction, reward
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Thực hiện một bước mô phỏng với hành động đã chọn.
        
        Args:
            action: Hành động được chọn (shard đích, giao thức đồng thuận)
        
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: Trạng thái mới, phần thưởng, trạng thái kết thúc, thông tin bổ sung
        """
        # Tăng số bước hiện tại
        self.current_step += 1
        
        # Tạo giao dịch mới
        new_transactions = self._generate_transactions()
        self.transaction_pool.extend(new_transactions)
        
        # Chọn một giao dịch để xử lý (ưu tiên các giao dịch có độ ưu tiên cao)
        pending_transactions = [tx for tx in self.transaction_pool if tx['status'] == 'pending']
        if not pending_transactions:
            # Nếu không có giao dịch đang chờ, trả về phần thưởng trung bình
            return self.get_state(), 0.0, self.current_step >= self.max_steps, {}
        
        # Sắp xếp giao dịch theo độ ưu tiên
        pending_transactions.sort(key=lambda tx: tx['priority'], reverse=True)
        current_transaction = pending_transactions[0]
        
        # Xử lý giao dịch với hành động đã chọn
        processed_transaction, reward = self._process_transaction(current_transaction, action)
        
        # Cập nhật transaction pool
        for i, tx in enumerate(self.transaction_pool):
            if tx['id'] == processed_transaction['id']:
                self.transaction_pool[i] = processed_transaction
                break
        
        # Giảm mức độ tắc nghẽn theo thời gian
        self.shard_congestion = np.maximum(0.0, self.shard_congestion - 0.02)
        
        # Cập nhật metrics
        completed_txs = [tx for tx in self.transaction_pool if tx['status'] == 'completed']
        if completed_txs:
            # Throughput: số giao dịch hoàn thành trong bước hiện tại
            throughput = len([tx for tx in completed_txs if tx['completion_time'] == self.current_step])
            self.metrics['throughput'].append(throughput)
            
            # Độ trễ trung bình
            latencies = [tx['completion_time'] - tx['created_at'] for tx in completed_txs]
            self.metrics['latency'].append(np.mean(latencies) if latencies else 0)
            
            # Tiêu thụ năng lượng trung bình
            energies = [tx['energy_consumed'] for tx in completed_txs]
            self.metrics['energy_consumption'].append(np.mean(energies) if energies else 0)
            
            # Đánh giá bảo mật (dựa trên % giao dịch sử dụng các giao thức đồng thuận mạnh)
            robust_rate = len([tx for tx in completed_txs if tx['consensus_protocol'] == 'Robust_BFT']) / len(completed_txs)
            pbft_rate = len([tx for tx in completed_txs if tx['consensus_protocol'] == 'PBFT']) / len(completed_txs)
            security_score = robust_rate + 0.7 * pbft_rate
            self.metrics['security_score'].append(security_score)
        
        # Kiểm tra điều kiện kết thúc
        done = self.current_step >= self.max_steps
        
        # Thông tin bổ sung
        info = {
            'transaction': processed_transaction,
            'metrics': {k: self.metrics[k][-1] if self.metrics[k] else 0 for k in self.metrics}
        }
        
        return self.get_state(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """Đặt lại môi trường về trạng thái ban đầu."""
        self.current_step = 0
        self.transaction_pool = []
        
        # Khởi tạo lại mạng blockchain
        self._init_blockchain_network()
        
        # Đặt lại metrics
        self.metrics = {
            'throughput': [],
            'latency': [],
            'energy_consumption': [],
            'security_score': []
        }
        
        return self.get_state()
    
    def render(self, mode='human'):
        """
        Hiển thị trạng thái hiện tại của môi trường.
        
        Args:
            mode: Chế độ render
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Number of transactions: {len(self.transaction_pool)}")
            print(f"Pending transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'pending'])}")
            print(f"Completed transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'completed'])}")
            print(f"Failed transactions: {len([tx for tx in self.transaction_pool if tx['status'] == 'failed'])}")
            
            if self.metrics['throughput']:
                print(f"Current throughput: {self.metrics['throughput'][-1]}")
            if self.metrics['latency']:
                print(f"Current average latency: {self.metrics['latency'][-1]:.2f}")
            if self.metrics['energy_consumption']:
                print(f"Current average energy consumption: {self.metrics['energy_consumption'][-1]:.2f}")
            if self.metrics['security_score']:
                print(f"Current security score: {self.metrics['security_score'][-1]:.2f}")
            
            print(f"Shard congestion: {', '.join([f'{c:.2f}' for c in self.shard_congestion])}")
            print("-" * 50)
    
    def close(self):
        """Đóng môi trường và giải phóng tài nguyên."""
        pass
    
    def get_congestion_level(self):
        """Lấy mức độ tắc nghẽn trung bình hiện tại của mạng."""
        return np.mean(self.shard_congestion)
    
    def get_congestion_data(self):
        """
        Lấy dữ liệu tắc nghẽn cho các shard.
        
        Returns:
            np.ndarray: Mảng chứa mức độ tắc nghẽn của mỗi shard
        """
        return self.shard_congestion.copy() 