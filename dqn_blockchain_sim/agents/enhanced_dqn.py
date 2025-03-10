"""
Module triển khai các mô hình DQN nâng cao cho việc tối ưu hóa blockchain
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import deque

from dqn_blockchain_sim.agents.dqn_agent import ReplayBuffer


class DuelingDQN(nn.Module):
    """
    Mạng Dueling DQN với kiến trúc tách biệt giá trị trạng thái và lợi thế hành động
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [128, 128]):
        """
        Khởi tạo mạng Dueling DQN
        
        Args:
            state_dim: Số chiều của không gian trạng thái
            action_dim: Số chiều của không gian hành động
            hidden_layers: Danh sách các nơ-ron trong mỗi lớp ẩn
        """
        super(DuelingDQN, self).__init__()
        
        # Lớp đặc trưng chung
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU()
        )
        
        # Lớp giá trị trạng thái
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[1] // 2, 1)
        )
        
        # Lớp lợi thế hành động
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[1] // 2, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua mạng
        
        Args:
            x: Tensor đầu vào biểu diễn trạng thái
            
        Returns:
            Tensor đầu ra biểu diễn Q-values
        """
        features = self.feature_layer(x)
        
        # Tính giá trị trạng thái
        value = self.value_stream(features)
        
        # Tính lợi thế hành động
        advantage = self.advantage_stream(features)
        
        # Kết hợp giá trị và lợi thế
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class NoisyLinear(nn.Module):
    """
    Lớp tuyến tính với nhiễu tham số cho việc khám phá
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        """
        Khởi tạo lớp tuyến tính nhiễu
        
        Args:
            in_features: Số đặc trưng đầu vào
            out_features: Số đặc trưng đầu ra
            std_init: Độ lệch chuẩn ban đầu cho nhiễu
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Tham số trọng số và độ lệch
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        # Khởi tạo tham số
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """
        Khởi tạo lại tham số
        """
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """
        Tạo lại nhiễu
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        Tạo nhiễu theo phân phối chuẩn
        
        Args:
            size: Kích thước của tensor nhiễu
            
        Returns:
            Tensor nhiễu đã được biến đổi
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua lớp
        
        Args:
            x: Tensor đầu vào
            
        Returns:
            Tensor đầu ra
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class NoisyDQN(nn.Module):
    """
    Mạng DQN với các lớp tuyến tính nhiễu
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: List[int] = [128, 128]):
        """
        Khởi tạo mạng Noisy DQN
        
        Args:
            state_dim: Số chiều của không gian trạng thái
            action_dim: Số chiều của không gian hành động
            hidden_layers: Danh sách các nơ-ron trong mỗi lớp ẩn
        """
        super(NoisyDQN, self).__init__()
        
        # Lớp đầu vào
        self.layers = nn.ModuleList([
            nn.Linear(state_dim, hidden_layers[0]),
            nn.ReLU()
        ])
        
        # Các lớp ẩn
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
            
        # Lớp đầu ra với nhiễu
        self.output_layer = NoisyLinear(hidden_layers[-1], action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass qua mạng
        
        Args:
            x: Tensor đầu vào biểu diễn trạng thái
            
        Returns:
            Tensor đầu ra biểu diễn Q-values
        """
        for layer in self.layers:
            x = layer(x)
            
        return self.output_layer(x)
    
    def reset_noise(self):
        """
        Đặt lại nhiễu cho tất cả các lớp nhiễu
        """
        self.output_layer.reset_noise()


class PrioritizedReplayBuffer:
    """
    Bộ nhớ trải nghiệm ưu tiên dựa trên TD-error
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Khởi tạo bộ nhớ trải nghiệm ưu tiên
        
        Args:
            capacity: Dung lượng tối đa của bộ nhớ
            alpha: Tham số alpha cho ưu tiên (0 = không ưu tiên, 1 = ưu tiên tối đa)
            beta: Tham số beta cho trọng số quan trọng (0 = không sửa chữa, 1 = sửa chữa đầy đủ)
            beta_increment: Tăng beta sau mỗi lần lấy mẫu
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Thêm trải nghiệm vào bộ nhớ
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Cờ đánh dấu kết thúc
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # Gán ưu tiên tối đa cho trải nghiệm mới
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Lấy mẫu một batch từ bộ nhớ dựa trên ưu tiên
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Tuple (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        # Tính xác suất lấy mẫu dựa trên ưu tiên
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Lấy mẫu dựa trên xác suất
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Tính trọng số quan trọng
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Tăng beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Lấy trải nghiệm
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Cập nhật ưu tiên cho các trải nghiệm
        
        Args:
            indices: Chỉ số của các trải nghiệm
            priorities: Ưu tiên mới
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """
        Trả về số lượng trải nghiệm trong bộ nhớ
        
        Returns:
            Số lượng trải nghiệm
        """
        return len(self.buffer)


class EnhancedDQNAgent:
    """
    Tác tử DQN nâng cao với nhiều cải tiến
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 network_type: str = 'dueling',
                 hidden_layers: List[int] = [128, 128],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update: int = 10,
                 prioritized_replay: bool = True,
                 double_dqn: bool = True):
        """
        Khởi tạo tác tử DQN nâng cao
        
        Args:
            state_dim: Số chiều của không gian trạng thái
            action_dim: Số chiều của không gian hành động
            network_type: Loại mạng ('standard', 'dueling', 'noisy')
            hidden_layers: Danh sách các nơ-ron trong mỗi lớp ẩn
            learning_rate: Tốc độ học
            gamma: Hệ số chiết khấu
            epsilon_start: Epsilon ban đầu cho chính sách epsilon-greedy
            epsilon_end: Epsilon tối thiểu
            epsilon_decay: Tốc độ giảm epsilon
            buffer_size: Kích thước bộ nhớ trải nghiệm
            batch_size: Kích thước batch
            target_update: Số bước cập nhật mạng mục tiêu
            prioritized_replay: Sử dụng bộ nhớ trải nghiệm ưu tiên
            double_dqn: Sử dụng Double DQN
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        
        # Khởi tạo thiết bị
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo mạng DQN
        if network_type == 'dueling':
            self.policy_net = DuelingDQN(state_dim, action_dim, hidden_layers).to(self.device)
            self.target_net = DuelingDQN(state_dim, action_dim, hidden_layers).to(self.device)
        elif network_type == 'noisy':
            self.policy_net = NoisyDQN(state_dim, action_dim, hidden_layers).to(self.device)
            self.target_net = NoisyDQN(state_dim, action_dim, hidden_layers).to(self.device)
        else:  # standard
            self.policy_net = nn.Sequential(
                nn.Linear(state_dim, hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[0], hidden_layers[1]),
                nn.ReLU(),
                nn.Linear(hidden_layers[1], action_dim)
            ).to(self.device)
            
            self.target_net = nn.Sequential(
                nn.Linear(state_dim, hidden_layers[0]),
                nn.ReLU(),
                nn.Linear(hidden_layers[0], hidden_layers[1]),
                nn.ReLU(),
                nn.Linear(hidden_layers[1], action_dim)
            ).to(self.device)
        
        # Sao chép trọng số từ policy net sang target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Đặt target net ở chế độ đánh giá
        
        # Khởi tạo bộ nhớ trải nghiệm
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = ReplayBuffer(buffer_size)
            
        # Khởi tạo optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Bộ đếm bước
        self.steps_done = 0
        
        # Lưu loại mạng và bộ nhớ
        self.network_type = network_type
        self.prioritized_replay = prioritized_replay
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Chọn hành động dựa trên trạng thái hiện tại
        
        Args:
            state: Trạng thái hiện tại
            
        Returns:
            Hành động được chọn
        """
        # Chuyển đổi state thành tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Nếu sử dụng mạng nhiễu, không cần epsilon-greedy
        if self.network_type == 'noisy':
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        
        # Epsilon-greedy
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_dim)
    
    def update_epsilon(self) -> None:
        """
        Cập nhật epsilon theo lịch trình
        """
        if self.network_type != 'noisy':  # Không cần epsilon cho mạng nhiễu
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def optimize_model(self) -> float:
        """
        Tối ưu hóa mô hình dựa trên batch trải nghiệm
        
        Returns:
            Giá trị mất mát
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Lấy mẫu từ bộ nhớ
        if self.prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            indices = None
            weights = None
            
        # Chuyển đổi sang tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Tính Q-values hiện tại
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Tính Q-values mục tiêu
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Chọn hành động từ policy net, đánh giá từ target net
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Vanilla DQN
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                
            # Tính Q-values mục tiêu
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Tính mất mát
        if self.prioritized_replay:
            # Tính TD-error
            td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
            
            # Cập nhật ưu tiên
            self.memory.update_priorities(indices, td_errors + 1e-6)  # Thêm epsilon nhỏ để tránh ưu tiên 0
            
            # Tính mất mát có trọng số
            loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        else:
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Tối ưu hóa
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        
        # Đặt lại nhiễu nếu sử dụng mạng nhiễu
        if self.network_type == 'noisy':
            self.policy_net.reset_noise()
            
        return loss.item()
    
    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """
        Cập nhật tác tử với trải nghiệm mới
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Cờ đánh dấu kết thúc
            
        Returns:
            Giá trị mất mát
        """
        # Thêm trải nghiệm vào bộ nhớ
        self.memory.add(state, action, reward, next_state, done)
        
        # Tối ưu hóa mô hình
        loss = self.optimize_model()
        
        # Cập nhật epsilon
        self.update_epsilon()
        
        # Cập nhật mạng mục tiêu
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Tăng bước
        self.steps_done += 1
        
        return loss
    
    def save_model(self, path: str) -> None:
        """
        Lưu mô hình
        
        Args:
            path: Đường dẫn lưu mô hình
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'network_type': self.network_type,
            'prioritized_replay': self.prioritized_replay,
            'double_dqn': self.double_dqn
        }, path)
    
    def load_model(self, path: str) -> None:
        """
        Tải mô hình
        
        Args:
            path: Đường dẫn tải mô hình
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.network_type = checkpoint['network_type']
        self.prioritized_replay = checkpoint['prioritized_replay']
        self.double_dqn = checkpoint['double_dqn'] 