"""
Module triển khai tác tử DQN (Deep Q-Network) cho việc tối ưu hóa blockchain
"""

import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import traceback

from dqn_blockchain_sim.configs.simulation_config import DQN_CONFIG

# Định nghĩa namedtuple cho trải nghiệm
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNetwork(nn.Module):
    """
    Mạng nơ-ron cho DQN
    """
    
    def __init__(self, state_dim: int, action_dim: int, layers: List[int]):
        """
        Khởi tạo mạng DQN
        
        Args:
            state_dim: Số chiều của không gian trạng thái
            action_dim: Số chiều của không gian hành động
            layers: Danh sách các nơ-ron trong mỗi lớp ẩn
        """
        super(DQNNetwork, self).__init__()
        
        # Xây dựng các lớp ẩn
        self.layers = nn.ModuleList()
        
        # Layer đầu vào
        self.layers.append(nn.Linear(state_dim, layers[0]))
        
        # Các lớp ẩn
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
        # Lớp đầu ra
        self.output_layer = nn.Linear(layers[-1], action_dim)
        
        # Cải thiện: Khởi tạo trọng số tốt hơn
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Khởi tạo trọng số theo phân phối tốt hơn"""
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0)
        
        # Khởi tạo lớp đầu ra với variance thấp hơn
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua mạng
        
        Args:
            x: Tensor đầu vào (batch_size, state_dim)
            
        Returns:
            Tensor đầu ra (batch_size, action_dim)
        """
        # Truyền qua các lớp ẩn với hàm kích hoạt ReLU
        for layer in self.layers:
            x = F.relu(layer(x))
            
        # Lớp đầu ra không có hàm kích hoạt
        return self.output_layer(x)


class ReplayBuffer:
    """
    Bộ nhớ trải nghiệm cho DQN
    """
    
    def __init__(self, capacity: int):
        """
        Khởi tạo bộ nhớ trải nghiệm
        
        Args:
            capacity: Dung lượng tối đa của bộ nhớ
        """
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """
        Thêm một trải nghiệm vào bộ nhớ
        
        Args:
            state: Trạng thái
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: Đã kết thúc (đúng/sai)
        """
        self.memory.append(Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Danh sách các trải nghiệm
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)


class ShardDQNAgent:
    """
    Tác tử DQN cho việc tối ưu hóa blockchain
    """
    
    def __init__(self, shard_id: int, state_size: int, action_size: int, config: Dict[str, Any] = None):
        """
        Khởi tạo tác tử DQN
        
        Args:
            shard_id: ID của shard
            state_size: Kích thước trạng thái
            action_size: Kích thước hành động
            config: Cấu hình cho DQN, dùng mặc định nếu không cung cấp
        """
        self.shard_id = shard_id
        self.state_size = state_size
        self.action_size = action_size
        self.config = config if config is not None else DQN_CONFIG.copy()
        
        # Thiết lập device (CPU hoặc GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("use_gpu", True) else "cpu")
        
        # Thiết lập mạng chính và mạng mục tiêu
        self.policy_net = DQNNetwork(
            state_dim=state_size,
            action_dim=action_size,
            layers=self.config.get("hidden_layers", [64, 64])
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            state_dim=state_size,
            action_dim=action_size,
            layers=self.config.get("hidden_layers", [64, 64])
        ).to(self.device)
        
        # Sao chép trọng số từ mạng chính sang mạng mục tiêu
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Đặt mạng mục tiêu trong chế độ đánh giá
        
        # Tạo bộ nhớ trải nghiệm
        self.memory = ReplayBuffer(self.config.get("memory_size", 10000))
        
        # Thiết lập bộ tối ưu hóa
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.get("learning_rate", 0.001)
        )
        
        # Giá trị epsilon cho chiến lược epsilon-greedy
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.05)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        
        # Hệ số chiết khấu (giảm giá)
        self.gamma = self.config.get("gamma", 0.99)
        
        # Chỉ huấn luyện mạng sau mỗi vài bước
        self.train_frequency = self.config.get("train_frequency", 4)
        
        # Cập nhật mạng mục tiêu sau mỗi vài bước
        self.target_update_frequency = self.config.get("target_update_frequency", 100)
        
        # Tổng số bước
        self.total_steps = 0
        
        # Lưu giữ lịch sử huấn luyện
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
        # Thiết lập clip gradients
        self.clip_grad = self.config.get("clip_grad", False)
        self.clip_value = self.config.get("clip_value", 1.0)
        
        # Khởi tạo trạng thái hiện tại
        self.current_state = None
        self.last_action = 0
        
        # Tốc độ học điều chỉnh
        self.adaptive_lr = self.config.get("adaptive_lr", False)
        self.min_lr = self.config.get("min_lr", 0.0001)
        
        # Double DQN
        self.use_double_dqn = self.config.get("use_double_dqn", True)
        
        # Ưu tiên mẫu (PER - Prioritized Experience Replay)
        self.use_per = self.config.get("use_per", False)
        
        # Huber loss cho training ổn định hơn
        self.use_huber_loss = self.config.get("use_huber_loss", True)
        
        # Cờ cho biết có đang trong quá trình huấn luyện hay không
        self.is_training = self.config.get("is_training", True)
        
        print(f"DQN Agent cho shard {shard_id} được khởi tạo thành công! (Device: {self.device})")
    
    def reset(self):
        """
        Đặt lại trạng thái của tác tử DQN
        """
        self.current_state = None
        self.last_action = 0
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.total_steps = 0
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        print(f"DQN Agent cho shard {self.shard_id} đã được reset!")

    def update(self, state, action, reward, next_state, done):
        """
        Cập nhật tác tử DQN với một bước trải nghiệm hoàn chỉnh
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp
            done: Đã kết thúc hay chưa
        """
        # Xử lý trạng thái đầu vào
        if state is not None:
            state = self.preprocess_state(state)
        else:
            state = np.zeros(self.state_size, dtype=np.float32)
            
        # Lưu trạng thái hiện tại
        self.current_state = state
        self.last_action = action
        
        # Xử lý trạng thái kế tiếp
        if next_state is not None:
            next_state = self.preprocess_state(next_state)
        else:
            next_state = np.zeros(self.state_size, dtype=np.float32)
            
        # Lưu trải nghiệm vào bộ nhớ
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )
        
        # Lưu phần thưởng vào lịch sử
        self.reward_history.append(reward)
        
        # Tăng số bước
        self.total_steps += 1
        
        # Giảm epsilon theo thời gian (epsilon decay)
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        # Lưu epsilon vào lịch sử
        self.epsilon_history.append(self.epsilon)
        
        # Huấn luyện mạng nếu đủ điều kiện
        if (len(self.memory) >= self.config.get("batch_size", 64) and
            self.total_steps % self.train_frequency == 0 and
            self.is_training):
            self.optimize_model(batch_size=self.config.get("batch_size", 64))
            
        # Cập nhật mạng mục tiêu nếu đủ điều kiện
        if self.total_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def preprocess_state(self, state):
        """
        Tiền xử lý trạng thái để đảm bảo tính nhất quán
        
        Args:
            state: Trạng thái đầu vào
            
        Returns:
            Trạng thái đã được xử lý
        """
        # Chuyển đổi sang numpy array nếu cần
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
            
        # Đảm bảo kích thước đúng
        if state.shape[0] < self.state_size:
            # Nếu thiếu chiều, pad thêm zeros
            padded_state = np.zeros(self.state_size, dtype=np.float32)
            padded_state[:state.shape[0]] = state
            state = padded_state
        elif state.shape[0] > self.state_size:
            # Nếu thừa chiều, chỉ lấy số chiều cần thiết
            state = state[:self.state_size]
            
        return state
        
    def predict(self, state):
        """
        Dự đoán giá trị Q cho từng hành động
        
        Args:
            state: Trạng thái hiện tại
            
        Returns:
            Mảng các giá trị Q cho mỗi hành động
        """
        if state is None:
            # Nếu không có trạng thái, tạo một trạng thái mặc định
            state = np.zeros(self.state_size, dtype=np.float32)
        
        # Đảm bảo state là numpy array
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
            
        # Kiểm tra kích thước trạng thái
        if state.shape[0] != self.state_size:
            # Xử lý state để đạt kích thước đúng
            state = self.preprocess_state(state)
            
        # Chuyển đổi sang tensor và đặt trên device đúng
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Đặt mạng trong chế độ đánh giá
        self.policy_net.eval()
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
        # Trả về kết quả dưới dạng numpy array
        return q_values.cpu().numpy()[0]
    
    def select_action(self, state=None):
        """
        Chọn hành động sử dụng chiến lược epsilon-greedy
        
        Args:
            state: Trạng thái hiện tại
            
        Returns:
            Hành động được chọn
        """
        if state is None:
            # Nếu không có trạng thái, tạo một trạng thái mặc định
            state = np.zeros(self.state_size, dtype=np.float32)
        
        # Lưu trạng thái hiện tại
        self.current_state = self.preprocess_state(state)
        
        # Epsilon-greedy: chọn ngẫu nhiên với xác suất epsilon
        if random.random() <= self.epsilon:
            # Khám phá: chọn hành động ngẫu nhiên
            action = random.randint(0, self.action_size - 1)
        else:
            # Khai thác: chọn hành động tốt nhất theo mạng
            q_values = self.predict(self.current_state)
            action = np.argmax(q_values)
            
        # Lưu lại hành động cuối cùng
        self.last_action = action
        
        return action
    
    def reward(self, reward, next_state=None, done=False):
        """
        Cập nhật tác tử DQN với phần thưởng nhận được
        
        Args:
            reward: Phần thưởng nhận được
            next_state: Trạng thái kế tiếp (tùy chọn)
            done: Đã kết thúc hay chưa (tùy chọn)
        """
        # Kiểm tra xem có trạng thái hiện tại không
        if self.current_state is None:
            print(f"Cảnh báo: Không có trạng thái hiện tại cho DQN Agent (shard {self.shard_id})")
            return
        
        # Lấy hành động cuối cùng đã thực hiện (mặc định là 0 nếu không có)
        action = getattr(self, 'last_action', 0)
            
        # Xử lý trạng thái kế tiếp
        if next_state is None:
            # Nếu không có trạng thái kế tiếp, giả định là giống trạng thái hiện tại
            next_state = self.current_state
        else:
            next_state = self.preprocess_state(next_state)
            
        # Lưu trải nghiệm vào bộ nhớ
        self.memory.push(
            self.current_state,
            action,
            reward,
            next_state,
            done
        )
        
        # Lưu lại hành động cuối cùng để sử dụng cho lần reward tiếp theo
        self.last_action = action
        
        # Cập nhật trạng thái hiện tại
        self.current_state = next_state
        
        # Lưu phần thưởng vào lịch sử
        self.reward_history.append(reward)
        
        # Tăng số bước
        self.total_steps += 1
        
        # Giảm epsilon theo thời gian (epsilon decay)
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        # Lưu epsilon vào lịch sử
        self.epsilon_history.append(self.epsilon)
        
        # Huấn luyện mạng nếu đủ điều kiện
        if (len(self.memory) >= self.config.get("batch_size", 64) and
            self.total_steps % self.train_frequency == 0):
            self.optimize_model(batch_size=self.config.get("batch_size", 64))
            
        # Cập nhật mạng mục tiêu nếu đủ điều kiện
        if self.total_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def optimize_model(self, batch_size):
        """
        Huấn luyện mạng DQN với một batch từ bộ nhớ trải nghiệm
        
        Args:
            batch_size: Kích thước batch
        """
        # Kiểm tra xem có đủ mẫu trong bộ nhớ không
        if len(self.memory) < max(batch_size, self.config.get("min_memory_size", 64)):
            return
            
        try:
            # Lấy một batch ngẫu nhiên từ bộ nhớ
            transitions = self.memory.sample(batch_size)
            
            # Chuyển đổi batch thành các tensor riêng biệt sử dụng zip(*) để unzip
            batch = Transition(*zip(*transitions))
            
            # Tạo mặt nạ cho các trạng thái không phải là terminal
            non_final_mask = torch.tensor(
                tuple(map(lambda d: not d, batch.done)),
                device=self.device,
                dtype=torch.bool
            )
            
            # Chỉ lấy các trạng thái kế tiếp không phải là terminal
            non_final_next_states = torch.tensor(
                [s for s, d in zip(batch.next_state, batch.done) if not d],
                device=self.device,
                dtype=torch.float32
            )
            
            # Tạo tensor cho trạng thái, hành động và phần thưởng
            state_batch = torch.tensor(
                batch.state,
                device=self.device,
                dtype=torch.float32
            )
            
            action_batch = torch.tensor(
                batch.action,
                device=self.device,
                dtype=torch.long
            ).unsqueeze(1)  # Thêm chiều cho gather
            
            reward_batch = torch.tensor(
                batch.reward,
                device=self.device,
                dtype=torch.float32
            )
            
            # Tính toán giá trị Q cho hành động đã thực hiện
            # gather(1, action_batch) lấy giá trị Q tương ứng với hành động đã chọn
            q_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # Tính toán giá trị Q mong đợi cho trạng thái kế tiếp
            next_state_values = torch.zeros(batch_size, device=self.device)
            
            # Double DQN: Sử dụng policy net để chọn hành động,
            # và target net để đánh giá giá trị của hành động đó
            if self.use_double_dqn:
                with torch.no_grad():
                    # Chọn hành động bằng policy net
                    next_action_indices = self.policy_net(non_final_next_states).max(1)[1].detach()
                    
                    # Đánh giá giá trị bằng target net
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(
                        1, next_action_indices.unsqueeze(1)
                    ).squeeze(1)
            else:
                # Standard DQN
                with torch.no_grad():
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
            
            # Tính toán giá trị Q mong đợi
            expected_q_values = reward_batch + (self.gamma * next_state_values)
            
            # Tính toán loss
            if self.use_huber_loss:
                # Huber loss cho robust learning
                loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))
            else:
                # MSE loss
                loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))
                
            # Tối ưu hóa mô hình
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients nếu cần
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
                
            self.optimizer.step()
            
            # Lưu loss vào lịch sử
            self.loss_history.append(loss.item())
            
            # Điều chỉnh tốc độ học nếu cần
            if self.adaptive_lr and len(self.loss_history) > 100:
                recent_losses = self.loss_history[-100:]
                if np.mean(recent_losses) < 0.01:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * 0.9, self.min_lr)
                        
            return loss.item()
        except Exception as e:
            print(f"Lỗi trong quá trình tối ưu hóa mô hình DQN: {e}")
            traceback.print_exc()
            return None
            
    def save(self, path):
        """
        Lưu mô hình
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'config': self.config,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'epsilon_history': self.epsilon_history
        }, path)
        
    def load(self, path):
        """
        Tải mô hình
        
        Args:
            path: Đường dẫn tải mô hình
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']
        self.config = checkpoint['config']
        self.loss_history = checkpoint['loss_history']
        self.reward_history = checkpoint['reward_history']
        self.epsilon_history = checkpoint['epsilon_history']
        
    def train(self, epochs=1, batch_size=None, save_path=None, save_frequency=100):
        """
        Huấn luyện mô hình
        
        Args:
            epochs: Số lượng epochs
            batch_size: Kích thước batch (sử dụng từ config nếu không được cung cấp)
            save_path: Đường dẫn lưu mô hình
            save_frequency: Tần suất lưu mô hình
            
        Returns:
            Lịch sử loss
        """
        # Sử dụng batch_size từ config nếu không được cung cấp
        if batch_size is None:
            batch_size = self.config.get("batch_size", 64)
            
        # Nếu không đủ mẫu, không huấn luyện
        if len(self.memory) < batch_size:
            return self.loss_history
            
        loss = None
        for epoch in range(epochs):
            loss = self.optimize_model(batch_size)
            
            if epoch % 100 == 0 and loss is not None:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}, Epsilon: {self.epsilon:.4f}")
                
            if save_path and epoch % save_frequency == 0:
                self.save(f"{save_path}_epoch_{epoch}.pt")
                
        if save_path:
            self.save(f"{save_path}_final.pt")
            
        return self.loss_history
        
    def get_statistics(self):
        """
        Lấy các thống kê của tác tử
        
        Returns:
            Từ điển chứa các thống kê
        """
        return {
            "shard_id": self.shard_id,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "epsilon_history": self.epsilon_history,
            "avg_recent_loss": np.mean(self.loss_history[-100:]) if len(self.loss_history) > 0 else 0,
            "avg_recent_reward": np.mean(self.reward_history[-100:]) if len(self.reward_history) > 0 else 0
        }


class MultiAgentDQNController:
    """
    Bộ điều khiển cho hệ thống đa tác tử DQN
    """
    
    def __init__(self, 
                 num_shards: int,
                 state_size: int,
                 action_size: int,
                 coordination_state_dim: int = None,
                 coordination_action_dim: int = None,
                 config: Dict[str, Any] = None):
        """
        Khởi tạo bộ điều khiển đa tác tử
        
        Args:
            num_shards: Số lượng shard/tác tử
            state_size: Số chiều của không gian trạng thái cho mỗi tác tử
            action_size: Số chiều của không gian hành động cho mỗi tác tử
            coordination_state_dim: Số chiều trạng thái cho tác tử phối hợp
            coordination_action_dim: Số chiều hành động cho tác tử phối hợp
            config: Cấu hình, sử dụng mặc định nếu không cung cấp
        """
        self.num_shards = num_shards
        self.config = config if config is not None else DQN_CONFIG
        
        # Tạo các tác tử DQN cho từng shard
        self.shard_agents = {}
        for i in range(num_shards):
            self.shard_agents[i] = ShardDQNAgent(i, state_size, action_size, self.config)
            
        # Tạo tác tử phối hợp (nếu có)
        self.has_coordinator = coordination_state_dim is not None and coordination_action_dim is not None
        
        if self.has_coordinator:
            self.coordination_agent = ShardDQNAgent(
                -1,  # ID đặc biệt cho tác tử phối hợp
                coordination_state_dim,
                coordination_action_dim,
                self.config
            )
        else:
            self.coordination_agent = None
            
        # Lưu trữ thông tin huấn luyện tổng thể
        self.global_steps = 0
        self.global_rewards = []
        self.global_losses = []
        
    def get_global_state(self, network_state: Dict[str, Any]) -> np.ndarray:
        """
        Lấy biểu diễn trạng thái cho tác tử phối hợp
        
        Args:
            network_state: Trạng thái toàn mạng
            
        Returns:
            Vector biểu diễn trạng thái toàn cục
        """
        if not self.has_coordinator:
            return None
            
        # Tạo vector trạng thái toàn cục từ thống kê mạng
        state = np.array([
            network_state['network_congestion'],  # Tắc nghẽn mạng
            network_state['avg_network_latency'] / 1000,  # Độ trễ mạng
            network_state['total_energy_consumption'] / 100,  # Tiêu thụ năng lượng
            network_state['cross_shard_transactions'] / network_state['total_transactions'] 
                if network_state['total_transactions'] > 0 else 0,  # Tỷ lệ giao dịch xuyên mảnh
            min([s['congestion_level'] for s in network_state['shard_stats'].values()]),  # Tắc nghẽn tối thiểu
            max([s['congestion_level'] for s in network_state['shard_stats'].values()]),  # Tắc nghẽn tối đa
            np.std([s['congestion_level'] for s in network_state['shard_stats'].values()]),  # Độ lệch chuẩn tắc nghẽn
            len(network_state['shard_stats']) / 20  # Số lượng shard (chuẩn hóa)
        ], dtype=np.float32)
        
        return state
    
    def select_coordination_action(self, state: np.ndarray) -> int:
        """
        Chọn hành động phối hợp dựa trên trạng thái toàn cục
        
        Args:
            state: Trạng thái toàn cục
            
        Returns:
            Hành động phối hợp được chọn
        """
        if not self.has_coordinator:
            return None
            
        return self.coordination_agent.select_action(state)
    
    def calculate_reward(self, 
                        shard_id: int, 
                        old_state: Dict[str, Any], 
                        new_state: Dict[str, Any],
                        network_state: Dict[str, Any]) -> float:
        """
        Tính toán phần thưởng cho một shard
        
        Args:
            shard_id: ID của shard
            old_state: Trạng thái cũ của shard
            new_state: Trạng thái mới của shard
            network_state: Trạng thái toàn mạng
            
        Returns:
            Giá trị phần thưởng
        """
        # Tính toán phần thưởng dựa trên nhiều yếu tố
        # 1. Thông lượng: Thưởng cho việc tăng số lượng giao dịch được xác nhận
        throughput_reward = (new_state['confirmed_transactions'] - old_state['confirmed_transactions']) * 0.01
        
        # 2. Độ trễ: Phạt cho độ trễ cao
        latency_reward = -new_state['avg_latency'] * 0.001
        
        # 3. Tắc nghẽn: Phạt cho tắc nghẽn cao
        congestion_reward = -new_state['congestion_level'] * 2.0
        
        # 4. Năng lượng: Phạt cho tiêu thụ năng lượng
        energy_reward = -(new_state['energy_consumption'] - old_state['energy_consumption']) * 0.01
        
        # Kết hợp với trọng số từ cấu hình
        total_reward = (
            self.config["reward_weights"]["throughput"] * throughput_reward +
            self.config["reward_weights"]["latency"] * latency_reward +
            self.config["reward_weights"]["security"] * congestion_reward +
            self.config["reward_weights"]["energy"] * energy_reward
        )
        
        return total_reward
    
    def calculate_coordination_reward(self, 
                                     old_network_state: Dict[str, Any], 
                                     new_network_state: Dict[str, Any]) -> float:
        """
        Tính toán phần thưởng cho tác tử phối hợp
        
        Args:
            old_network_state: Trạng thái mạng cũ
            new_network_state: Trạng thái mạng mới
            
        Returns:
            Giá trị phần thưởng
        """
        if not self.has_coordinator:
            return 0.0
            
        # Tính toán phần thưởng dựa trên hiệu suất toàn mạng
        
        # 1. Tắc nghẽn toàn mạng
        congestion_reward = -new_network_state['network_congestion'] * 5.0
        
        # 2. Độ lệch chuẩn tắc nghẽn (thưởng cho việc cân bằng tải)
        congestion_std = np.std([s['congestion_level'] for s in new_network_state['shard_stats'].values()])
        balance_reward = -congestion_std * 10.0
        
        # 3. Hiệu suất giao dịch xuyên mảnh
        cross_shard_ratio = (new_network_state['cross_shard_transactions'] / 
                            max(1, new_network_state['total_transactions']))
        cross_shard_efficiency = (new_network_state['cross_shard_transactions'] - 
                                old_network_state['cross_shard_transactions'])
        cross_shard_reward = cross_shard_efficiency * 0.1 - cross_shard_ratio * 2.0
        
        # 4. Tổng tiêu thụ năng lượng
        energy_diff = (new_network_state['total_energy_consumption'] - 
                      old_network_state['total_energy_consumption'])
        energy_reward = -energy_diff * 0.001
        
        # Tổng hợp phần thưởng
        total_reward = congestion_reward + balance_reward + cross_shard_reward + energy_reward
        
        return total_reward
    
    def train_agents(self, batch_size: int) -> Dict[str, float]:
        """
        Huấn luyện tất cả các tác tử
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Từ điển chứa các giá trị mất mát
        """
        losses = {}
        
        # Huấn luyện tác tử mỗi shard
        for shard_id, agent in self.shard_agents.items():
            loss = agent.optimize_model(batch_size)
            agent.update_epsilon()
            losses[f"shard_{shard_id}"] = loss
            
            # Cập nhật mạng mục tiêu nếu đến thời điểm
            if self.global_steps % self.config["target_update_freq"] == 0:
                agent.update_target_network()
                
        # Huấn luyện tác tử phối hợp (nếu có)
        if self.has_coordinator:
            coord_loss = self.coordination_agent.optimize_model(batch_size)
            self.coordination_agent.update_epsilon()
            losses["coordinator"] = coord_loss
            
            # Cập nhật mạng mục tiêu cho tác tử phối hợp
            if self.global_steps % self.config["target_update_freq"] == 0:
                self.coordination_agent.update_target_network()
                
        self.global_steps += 1
        self.global_losses.append(sum(losses.values()) / len(losses))
        
        return losses
    
    def save_all_agents(self, base_path: str) -> None:
        """
        Lưu tất cả các tác tử
        
        Args:
            base_path: Đường dẫn cơ sở để lưu mô hình
        """
        # Lưu tác tử mỗi shard
        for shard_id, agent in self.shard_agents.items():
            agent.save_model(f"{base_path}/shard_agent_{shard_id}.pt")
            
        # Lưu tác tử phối hợp (nếu có)
        if self.has_coordinator:
            self.coordination_agent.save_model(f"{base_path}/coordination_agent.pt")
            
    def load_all_agents(self, base_path: str) -> None:
        """
        Tải tất cả các tác tử
        
        Args:
            base_path: Đường dẫn cơ sở để tải mô hình
        """
        # Tải tác tử mỗi shard
        for shard_id, agent in self.shard_agents.items():
            try:
                agent.load_model(f"{base_path}/shard_agent_{shard_id}.pt")
            except FileNotFoundError:
                print(f"Không tìm thấy mô hình cho shard {shard_id}, sử dụng mô hình mới")
                
        # Tải tác tử phối hợp (nếu có)
        if self.has_coordinator:
            try:
                self.coordination_agent.load_model(f"{base_path}/coordination_agent.pt")
            except FileNotFoundError:
                print("Không tìm thấy mô hình cho tác tử phối hợp, sử dụng mô hình mới")
                
    def get_all_model_parameters(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Lấy tham số của tất cả các mô hình cho học liên kết
        
        Returns:
            Từ điển chứa tham số của tất cả các mô hình
        """
        parameters = {}
        
        # Lấy tham số của tác tử mỗi shard
        for shard_id, agent in self.shard_agents.items():
            parameters[f"shard_{shard_id}"] = agent.get_model_parameters()
            
        # Lấy tham số của tác tử phối hợp (nếu có)
        if self.has_coordinator:
            parameters["coordinator"] = self.coordination_agent.get_model_parameters()
            
        return parameters
    
    def set_model_parameters(self, parameters: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """
        Đặt tham số mô hình từ bên ngoài (dùng trong học liên kết)
        
        Args:
            parameters: Từ điển chứa tham số của tất cả các mô hình
        """
        # Đặt tham số cho tác tử mỗi shard
        for shard_id, agent in self.shard_agents.items():
            if f"shard_{shard_id}" in parameters:
                agent.set_model_parameters(parameters[f"shard_{shard_id}"])
                
        # Đặt tham số cho tác tử phối hợp (nếu có)
        if self.has_coordinator and "coordinator" in parameters:
            self.coordination_agent.set_model_parameters(parameters["coordinator"]) 