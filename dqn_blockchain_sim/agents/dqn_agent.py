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
            
        # Lớp đầu ra không có hàm kích hoạt (để ước tính giá trị Q)
        return self.output_layer(x)


class ReplayBuffer:
    """
    Bộ nhớ lưu lại trải nghiệm (experience replay) cho DQN
    """
    
    def __init__(self, capacity: int):
        """
        Khởi tạo bộ nhớ
        
        Args:
            capacity: Kích thước tối đa của bộ nhớ
        """
        self.memory = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
            next_state: Optional[np.ndarray], done: bool) -> None:
        """
        Thêm một trải nghiệm vào bộ nhớ
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo (None nếu là trạng thái kết thúc)
            done: Cờ kết thúc episode
        """
        self.memory.append(Transition(state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """
        Lấy mẫu ngẫu nhiên từ bộ nhớ
        
        Args:
            batch_size: Kích thước batch
            
        Returns:
            Danh sách các Transition
        """
        return random.sample(self.memory, min(batch_size, len(self.memory)))
        
    def __len__(self) -> int:
        """
        Lấy kích thước hiện tại của bộ nhớ
        
        Returns:
            Số lượng trải nghiệm trong bộ nhớ
        """
        return len(self.memory)


class ShardDQNAgent:
    """
    Tác tử DQN cho từng shard
    """
    
    def __init__(self, 
                 shard_id: int,
                 state_size: int,
                 action_size: int,
                 config: Dict[str, Any] = None):
        """
        Khởi tạo tác tử DQN cho shard
        
        Args:
            shard_id: ID của shard tác tử quản lý
            state_size: Số chiều của không gian trạng thái
            action_size: Số chiều của không gian hành động
            config: Cấu hình DQN, sử dụng mặc định nếu không cung cấp
        """
        self.shard_id = shard_id
        self.state_size = state_size
        self.action_size = action_size
        self.config = config if config is not None else DQN_CONFIG
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Khởi tạo epsilon cho chiến lược ε-greedy
        self.epsilon = self.config["epsilon_start"]
        self.epsilon_start = self.config["epsilon_start"]
        self.epsilon_end = self.config["epsilon_end"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.gamma = self.config["gamma"]  # Hệ số chiết khấu
        
        # Bộ nhớ replay
        self.memory = ReplayBuffer(self.config["replay_buffer_size"])
        
        # Bước thực hiện (dùng để giảm epsilon)
        self.steps_done = 0
        
        # Mạng chính sách và mạng đích
        self.policy_net = DQNNetwork(
            state_size, 
            action_size, 
            self.config["local_network_architecture"]
        ).to(self.device)
        
        self.target_net = DQNNetwork(
            state_size, 
            action_size, 
            self.config["local_network_architecture"]
        ).to(self.device)
        
        # Sao chép tham số từ policy_net sang target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Đặt target_net ở chế độ evaluation
        
        # Định nghĩa optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config["learning_rate"]
        )
        
        # Lưu tham chiếu đến shard
        self.shard = None
        
        # Thông tin theo dõi quá trình huấn luyện
        self.training_info = {
            'episodes': 0,
            'rewards': [],
            'losses': [],
            'episode_lengths': []
        }
        
        # Thêm trạng thái hiện tại để theo dõi
        self.current_state = None
        self.last_action = None
        self.pending_reward = 0.0
        self.pending_experience = None
        
        # Theo dõi điểm thưởng tích lũy
        self.accumulated_reward = 0.0
        self.episode_actions = []
        
        # Cờ chỉ ra trạng thái huấn luyện
        self.is_training = True
        
    def get_state_representation(self, shard_state: Dict[str, Any]) -> np.ndarray:
        """
        Chuyển đổi trạng thái shard thành biểu diễn vector cho DQN
        
        Args:
            shard_state: Trạng thái của shard
            
        Returns:
            Vector biểu diễn trạng thái
        """
        # Vector hóa trạng thái của shard
        # Các đặc trưng quan trọng: tải, số nút, hiệu suất, v.v.
        state = np.array([
            shard_state['congestion_level'],  # Mức độ tắc nghẽn
            shard_state['node_count'] / 100,  # Số lượng nút (được chuẩn hóa)
            shard_state['pending_transactions'] / 1000,  # Số lượng giao dịch đang chờ
            shard_state['avg_latency'] / 1000,  # Độ trễ trung bình (ms)
            shard_state['energy_consumption'] / 10,  # Tiêu thụ năng lượng
            shard_state['avg_load']  # Tải trung bình
        ], dtype=np.float32)
        
        return state
    
    def predict(self, state: np.ndarray) -> int:
        """
        Dự đoán hành động tốt nhất cho trạng thái đã cho mà không sử dụng ε-greedy
        
        Args:
            state: Trạng thái hiện tại
            
        Returns:
            Hành động tốt nhất theo model
        """
        # Chuyển đổi state thành numpy array nếu nó là list
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # Đảm bảo state có kích thước đúng
        if not hasattr(state, 'shape'):
            print(f"Cảnh báo: State không có thuộc tính shape. Loại: {type(state)}")
            # Chuyển đổi thành numpy array
            state = np.array(state, dtype=np.float32)
            
        if state.shape[0] != self.state_size:
            print(f"Cảnh báo: Kích thước state không khớp. Nhận được {state.shape[0]}, cần {self.state_size}")
            # Điều chỉnh kích thước state
            if state.shape[0] < self.state_size:
                # Nếu state nhỏ hơn, thêm các giá trị 0
                padded_state = np.zeros(self.state_size)
                padded_state[:state.shape[0]] = state
                state = padded_state
            else:
                # Nếu state lớn hơn, cắt bớt
                state = state[:self.state_size]
        
        # Chuyển đổi state sang tensor và thêm batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Dự đoán hành động tốt nhất
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()
    
    def reward(self, reward_value: float) -> None:
        """
        Cung cấp phần thưởng cho agent sau khi thực hiện hành động
        
        Args:
            reward_value: Giá trị phần thưởng
        """
        # Cập nhật phần thưởng tích lũy
        self.accumulated_reward += reward_value
        self.pending_reward += reward_value
        
        # Nếu có trải nghiệm đang chờ xử lý, cập nhật nó
        if self.pending_experience is not None:
            state, action, _, next_state, done = self.pending_experience
            self.pending_experience = (state, action, self.pending_reward, next_state, done)
            
            # Thêm vào bộ nhớ nếu đã đủ thông tin
            if next_state is not None:
                self.add_experience(*self.pending_experience)
                self.pending_experience = None
                self.pending_reward = 0.0
    
    def train(self, batch_size: int = None) -> float:
        """
        Huấn luyện agent từ bộ nhớ replay
        
        Args:
            batch_size: Kích thước batch, mặc định lấy từ config
            
        Returns:
            Giá trị loss sau khi huấn luyện
        """
        # Nếu không ở chế độ huấn luyện, không làm gì cả
        if not self.is_training:
            return 0.0
            
        # Sử dụng batch_size từ config nếu không cung cấp
        if batch_size is None:
            batch_size = self.config['batch_size']
            
        # Tối ưu hóa mô hình
        loss = self.optimize_model(batch_size)
        
        # Cập nhật mạng mục tiêu sau mỗi số bước nhất định
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.update_target_network()
            
        # Cập nhật epsilon
        self.update_epsilon()
        
        return loss

    def select_action(self, state: np.ndarray = None) -> int:
        """
        Chọn hành động dựa trên trạng thái hiện tại
        
        Args:
            state: Trạng thái hiện tại, hoặc None để lấy trạng thái từ shard
            
        Returns:
            Hành động được chọn (index trong không gian hành động)
        """
        # Nếu state không được cung cấp, lấy từ shard được gán
        if state is None:
            if self.shard is not None and hasattr(self.shard, 'get_state'):
                state = self.shard.get_state()
            else:
                # Nếu không có shard hoặc shard không có phương thức get_state,
                # tạo một trạng thái mặc định
                state = np.zeros(self.state_size)
        
        # Chuyển đổi state thành numpy array nếu nó là list
        if isinstance(state, list):
            state = np.array(state, dtype=np.float32)
        
        # Đảm bảo state có kích thước đúng
        if not hasattr(state, 'shape'):
            print(f"Cảnh báo: State không có thuộc tính shape. Loại: {type(state)}")
            # Chuyển đổi thành numpy array
            state = np.array(state, dtype=np.float32)
            
        if state.shape[0] != self.state_size:
            print(f"Cảnh báo: Kích thước state không khớp. Nhận được {state.shape[0]}, cần {self.state_size}")
            # Điều chỉnh kích thước state
            if state.shape[0] < self.state_size:
                # Nếu state nhỏ hơn, thêm các giá trị 0
                padded_state = np.zeros(self.state_size)
                padded_state[:state.shape[0]] = state
                state = padded_state
            else:
                # Nếu state lớn hơn, cắt bớt
                state = state[:self.state_size]
        
        # Chuyển đổi state sang tensor và thêm batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Lưu trạng thái hiện tại cho các cập nhật trong tương lai
        self.current_state = state
        
        # Quyết định xem có thực hiện khám phá hay không
        sample = random.random()
        
        # Nếu các thuộc tính epsilon_* không tồn tại, sử dụng epsilon trực tiếp
        if hasattr(self, 'epsilon_start') and hasattr(self, 'epsilon_end') and hasattr(self, 'epsilon_decay'):
            eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                math.exp(-1. * self.steps_done / self.epsilon_decay)
        else:
            eps_threshold = self.epsilon
        
        self.steps_done += 1
        
        # Lựa chọn hành động
        if sample > eps_threshold and not self.is_training:
            # Khi đánh giá, không sử dụng khám phá ngẫu nhiên
            with torch.no_grad():
                action = self.policy_net(state_tensor).max(1)[1].item()
        elif sample > eps_threshold:
            # Trong quá trình huấn luyện, sử dụng mô hình để chọn
            with torch.no_grad():
                action = self.policy_net(state_tensor).max(1)[1].item()
        else:
            # Chọn hành động ngẫu nhiên
            action = random.randrange(self.action_size)
        
        # Lưu hành động cuối cùng
        self.last_action = action
        self.episode_actions.append(action)
        
        # Cập nhật trải nghiệm đang chờ
        if self.current_state is not None and self.last_action is not None:
            # Tạo trải nghiệm chờ (state, action, reward=0, next_state=None, done=False)
            self.pending_experience = (self.current_state, self.last_action, 0, None, False)
        
        return action
        
    def update_epsilon(self) -> None:
        """
        Cập nhật epsilon theo lịch trình
        """
        self.epsilon = max(
            self.config["epsilon_end"],
            self.epsilon - self.config["epsilon_decay"]
        )
        
    def optimize_model(self, batch_size: int) -> float:
        """
        Cập nhật mô hình từ bộ nhớ
        
        Args:
            batch_size: Số lượng mẫu để xử lý
            
        Returns:
            Giá trị loss
        """
        # Kiểm tra xem bộ nhớ có đủ mẫu không
        if len(self.memory) < batch_size:
            return 0.0
        
        # Lấy mẫu ngẫu nhiên từ bộ nhớ
        transitions = self.memory.sample(batch_size)
        
        # Chuyển đổi batch thành các tensor riêng biệt
        batch = Transition(*zip(*transitions))
        
        # Tạo mask cho các trạng thái không phải kết thúc
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device, 
            dtype=torch.bool
        )
        
        # Đếm số lượng trạng thái tiếp theo không phải kết thúc
        non_final_count = non_final_mask.sum().item()
        
        # Nếu không có trạng thái tiếp theo nào, trả về 0
        if non_final_count == 0:
            return 0.0
        
        # Lọc các trạng thái tiếp theo không phải kết thúc
        non_final_next_states = torch.FloatTensor(
            [s for s in batch.next_state if s is not None]
        ).to(self.device)
        
        # Đảm bảo kích thước đúng cho các tensor
        state_batch = torch.FloatTensor([s for s in batch.state]).to(self.device)
        action_batch = torch.LongTensor([a for a in batch.action]).to(self.device)
        reward_batch = torch.FloatTensor([r for r in batch.reward]).to(self.device)
        
        # Tính Q-values cho các hành động đã thực hiện
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Tính giá trị Q cho trạng thái tiếp theo
        next_state_values = torch.zeros(batch_size, device=self.device)
        
        # Sử dụng Double DQN để giảm thiểu overestimation bias
        if self.config.get("use_double_dqn", False) and non_final_count > 0:
            # Chọn hành động từ policy network
            with torch.no_grad():
                next_action_indices = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                # Đánh giá giá trị từ target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_action_indices).squeeze(1)
        elif non_final_count > 0:
            # Vanilla DQN
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Tính giá trị Q mục tiêu
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # Tính loss
        if self.config.get("use_huber_loss", True):
            # Huber loss cho robust learning
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        else:
            # MSE loss
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Tối ưu hóa mô hình
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        if self.config.get("use_gradient_clipping", True):
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.get("gradient_clip_value", 1.0))
        
        self.optimizer.step()
        
        # Lưu giá trị loss
        loss_value = loss.item()
        self.training_info['losses'].append(loss_value)
        
        return loss_value
        
    def add_experience(self, state, action, reward, next_state, done):
        """
        Thêm trải nghiệm vào bộ nhớ replay
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Cờ đánh dấu kết thúc episode
        """
        # Chuyển next_state thành None nếu episode kết thúc
        if done:
            next_state = None
            
        # Thêm vào bộ nhớ
        self.memory.push(state, action, reward, next_state, done)
        
        # Cập nhật thông tin huấn luyện
        self.training_info['rewards'].append(reward)
        
        # Nếu episode kết thúc, cập nhật thông tin episode
        if done:
            self.training_info['episodes'] += 1
            self.training_info['episode_lengths'].append(len(self.episode_actions))
            # Reset các biến theo dõi episode
            self.accumulated_reward = 0.0
            self.episode_actions = []
            
    def update_target_network(self):
        """
        Cập nhật mạng mục tiêu từ mạng chính sách
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
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
            'training_info': self.training_info
        }, path)
        
    def load_model(self, path: str) -> None:
        """
        Tải mô hình
        
        Args:
            path: Đường dẫn đến mô hình
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.training_info = checkpoint['training_info']
        
    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Lấy tham số của mô hình để chia sẻ trong quá trình học liên kết
        
        Returns:
            Từ điển chứa các tham số mô hình
        """
        return {name: param.clone().detach() for name, param in self.policy_net.named_parameters()}
        
    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Đặt các tham số mô hình từ bên ngoài
        
        Args:
            parameters: Từ điển ánh xạ từ tên tham số đến giá trị tensor
        """
        # Kiểm tra các khóa có khớp không
        model_dict = self.policy_net.state_dict()
        to_load = {k: v for k, v in parameters.items() if k in model_dict}
        
        # Kiểm tra thiếu khóa
        missing = set(model_dict.keys()) - set(to_load.keys())
        if missing:
            print(f"Warning: Missing keys in parameters: {missing}")
            
        # Nạp tham số
        self.policy_net.load_state_dict(to_load, strict=False)
        self.update_target_network()
        
    def reset(self):
        """
        Đặt lại trạng thái của tác tử
        """
        # Đặt lại epsilon về giá trị ban đầu
        self.epsilon = self.epsilon_start
        
        # Đặt lại các biến theo dõi
        self.current_state = None
        self.last_action = None
        self.pending_reward = 0.0
        self.pending_experience = None
        self.accumulated_reward = 0.0
        self.episode_actions = []
        
        # Đặt lại thông tin huấn luyện
        self.training_info = {
            'episodes': 0,
            'rewards': [],
            'losses': [],
            'episode_lengths': []
        }
        
        # Đặt lại bước thực hiện
        self.steps_done = 0

    def update(self, state, action, reward, next_state, done):
        """
        Cập nhật tác tử với trải nghiệm mới
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Cờ đánh dấu kết thúc episode
            
        Returns:
            Giá trị loss từ quá trình cập nhật
        """
        # Thêm trải nghiệm vào bộ nhớ
        self.add_experience(state, action, reward, next_state, done)
        
        # Cập nhật epsilon
        self.update_epsilon()
        
        # Tối ưu hóa mô hình
        loss = self.optimize_model(self.config['batch_size'])
        
        # Cập nhật mạng mục tiêu sau mỗi số bước nhất định
        if self.steps_done % self.config['target_update_freq'] == 0:
            self.update_target_network()
            
        return loss


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