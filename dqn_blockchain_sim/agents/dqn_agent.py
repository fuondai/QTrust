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
        self.batch_norms = nn.ModuleList()
        
        # Layer đầu vào
        self.layers.append(nn.Linear(state_dim, layers[0]))
        self.batch_norms.append(nn.BatchNorm1d(layers[0]))
        
        # Các lớp ẩn
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            self.batch_norms.append(nn.BatchNorm1d(layers[i+1]))
            
        # Lớp đầu ra
        self.output_layer = nn.Linear(layers[-1], action_dim)
        
        # Cải thiện: Khởi tạo trọng số tốt hơn
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Khởi tạo trọng số theo phân phối tốt hơn"""
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.01)  # Thêm bias nhỏ dương để tránh neurons chết
        
        # Khởi tạo lớp đầu ra với variance thấp hơn để ổn định trong giai đoạn đầu
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
        # Xử lý batch size = 1 cho batch normalization
        single_input = x.dim() == 1 or x.size(0) == 1
        if single_input:
            x = x.unsqueeze(0) if x.dim() == 1 else x
            
        # Kiểm tra và điều chỉnh kích thước đầu vào
        input_dim = self.layers[0].in_features
        if x.shape[-1] != input_dim:
            # Nếu kích thước không khớp, cố gắng điều chỉnh
            if x.shape[-1] < input_dim:
                # Nếu đầu vào nhỏ hơn, thêm cột 0
                padding = torch.zeros(*x.shape[:-1], input_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Nếu đầu vào lớn hơn, cắt bớt
                x = x[..., :input_dim]
            
        # Truyền qua các lớp ẩn với hàm kích hoạt ReLU và batch normalization
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            # Chỉ áp dụng batch norm khi đào tạo với batch lớn hơn 1
            if not single_input or x.size(0) > 1:
                x = batch_norm(x)
            x = F.relu(x)
            
        # Lớp đầu ra không có hàm kích hoạt
        x = self.output_layer(x)
        
        # Khôi phục kích thước ban đầu nếu đầu vào là single
        if single_input and x.size(0) == 1:
            x = x.squeeze(0)
            
        return x


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
            
        if next_state is not None:
            next_state = self.preprocess_state(next_state)
        else:
            next_state = np.zeros(self.state_size, dtype=np.float32)
        
        # Chuyển đổi thành tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.FloatTensor([float(done)]).to(self.device)
        
        # Lưu trữ kinh nghiệm vào bộ nhớ
        self.memory.push(state, action, reward, next_state, done)
        
        # Nếu không có đủ mẫu hoặc không phải thời điểm đào tạo, thoát
        if len(self.memory) < self.config.get("batch_size", 32) or not self.is_training:
            return
        
        if self.total_steps % self.train_frequency != 0:
            self.total_steps += 1
            return
        
        # Lấy mẫu từ bộ nhớ
        batch_size = self.config.get("batch_size", 32)
        transitions = self.memory.sample(batch_size)
        
        # Chuyển đổi batch
        batch = Transition(*zip(*transitions))
        
        # Xử lý dữ liệu batch
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor([float(d) for d in batch.done]).to(self.device)
        
        # Tính toán Q-value cho hành động đã thực hiện
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Tính toán Q-value tốt nhất cho trạng thái tiếp theo
        if self.use_double_dqn:
            # Double DQN: Sử dụng policy net để chọn hành động, target net để đánh giá
            with torch.no_grad():
                next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
        else:
            # Vanilla DQN
            with torch.no_grad():
                next_state_values = self.target_net(next_state_batch).max(1)[0]
        
        # Tính toán giá trị mục tiêu
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Tính toán loss
        if self.use_huber_loss:
            # Huber loss cho khả năng chống nhiễu tốt hơn
            loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_state_action_values)
        else:
            # MSE loss
            loss = F.mse_loss(state_action_values.squeeze(1), expected_state_action_values)
        
        # Tối ưu hóa mô hình
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients nếu được cấu hình
        if self.clip_grad:
            nn.utils.clip_grad_value_(self.policy_net.parameters(), self.clip_value)
        
        self.optimizer.step()
        
        # Ghi lại loss
        self.loss_history.append(loss.item())
        
        # Cập nhật mạng mục tiêu
        if self.total_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Giảm epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        
        # Cập nhật tổng số bước
        self.total_steps += 1
        
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
        Dự đoán giá trị Q cho trạng thái đầu vào
        
        Args:
            state: Trạng thái đầu vào
            
        Returns:
            Mảng giá trị Q cho mỗi hành động
        """
        # Chuyển đổi state thành tensor
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).to(self.device)
        else:
            state_tensor = state.clone().detach().to(self.device)
            
        # Đảm bảo state_tensor có đúng kích thước
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Đặt mạng ở chế độ đánh giá
        self.policy_net.eval()
        
        # Dự đoán giá trị Q với torch.no_grad() để không tính gradient
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()
            
        # Chuyển về chế độ huấn luyện nếu đang huấn luyện
        if self.is_training:
            self.policy_net.train()
            
        return q_values[0] if len(q_values.shape) > 1 else q_values
    
    def act(self, state):
        """
        Chọn hành động dựa trên trạng thái đầu vào.
        Đây là phương thức chính để lựa chọn hành động trong quá trình mô phỏng.
        
        Args:
            state: Trạng thái đầu vào, có thể là array hoặc tensor
            
        Returns:
            Hành động được chọn
        """
        try:
            # Tiền xử lý trạng thái nếu cần
            if not isinstance(state, torch.Tensor) and not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
                
            # Đảm bảo kích thước trạng thái đúng với state_size
            if isinstance(state, np.ndarray):
                if len(state.shape) == 1:
                    # Trường hợp vector 1D
                    if state.shape[0] < self.state_size:
                        # Nếu nhỏ hơn, thêm các số 0
                        padding = np.zeros(self.state_size - state.shape[0], dtype=np.float32)
                        state = np.concatenate([state, padding])
                    elif state.shape[0] > self.state_size:
                        # Nếu lớn hơn, cắt bớt
                        state = state[:self.state_size]
                        
            # Sử dụng phương thức select_action để chọn hành động theo chiến lược epsilon-greedy
            return self.select_action(state)
        except Exception as e:
            print(f"Lỗi trong phương thức act(): {e}")
            # Trả về hành động mặc định trong trường hợp lỗi
            return 0
    
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
            
    def optimize_model(self, batch_size=None):
        """
        Huấn luyện mô hình DQN sử dụng một batch kinh nghiệm
        
        Args:
            batch_size: Kích thước batch để huấn luyện, mặc định sử dụng giá trị từ config
            
        Returns:
            float: Giá trị loss
        """
        if not self.memory:
            return 0.0
            
        # Sử dụng kích thước batch được cung cấp hoặc lấy từ config
        if batch_size is None:
            batch_size = self.config["batch_size"]
            
        # Đảm bảo có đủ dữ liệu trong bộ nhớ
        if len(self.memory) < batch_size:
            return 0.0
            
        # Lấy ngẫu nhiên một batch kinh nghiệm từ bộ nhớ
        transitions = self.memory.sample(batch_size)
        
        # Chuyển đổi batch thành các tensor riêng biệt
        batch = Transition(*zip(*transitions))
        
        # Tạo mask cho các trạng thái không phải là trạng thái kết thúc
        non_final_mask = torch.tensor([not done for done in batch.done], 
                                    dtype=torch.bool, device=self.device)
        
        # Lọc các trạng thái tiếp theo không phải là trạng thái kết thúc
        non_final_next_states = torch.cat([torch.FloatTensor([s]) for s, done 
                                        in zip(batch.next_state, batch.done) if not done]
                                       ).to(self.device)
        
        # Chuyển đổi các tensor khác
        state_batch = torch.cat([torch.FloatTensor([s]) for s in batch.state]).to(self.device)
        action_batch = torch.cat([torch.LongTensor([[a]]) for a in batch.action]).to(self.device)
        reward_batch = torch.cat([torch.FloatTensor([[r]]) for r in batch.reward]).to(self.device)
        
        # CẢI TIẾN 1: Áp dụng BATCH NORMALIZATION cho state_batch
        if hasattr(self, 'input_normalizer') and self.input_normalizer is not None:
            # Chuẩn hóa state batch
            with torch.no_grad():
                # Thêm batch normalizer để ổn định đầu vào
                state_batch = self.input_normalizer(state_batch)
                if len(non_final_next_states) > 0:
                    non_final_next_states = self.input_normalizer(non_final_next_states)
        
        # Tính toán Q(s_t, a) cho tất cả các cặp state, action trong batch
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Tính toán V(s_{t+1}) cho tất cả các trạng thái tiếp theo không phải kết thúc
        next_state_values = torch.zeros(batch_size, device=self.device)
        
        # CẢI TIẾN 2: Áp dụng DOUBLE DQN
        # Thay vì sử dụng max của target network, chúng ta chọn hành động tối ưu
        # từ policy network và đánh giá nó bằng target network
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                # Chọn hành động tốt nhất từ policy network
                if self.use_double_dqn:
                    # Lấy hành động từ policy network (Double DQN)
                    next_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    # Đánh giá giá trị Q của hành động đó bằng target network
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
                else:
                    # Lấy giá trị Q tối đa trực tiếp từ target network (DQN thông thường)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        
        # Tính giá trị Q mục tiêu: r + gamma * V(s_{t+1})
        expected_state_action_values = reward_batch + (self.gamma * next_state_values).unsqueeze(1)
        
        # CẢI TIẾN 3: Áp dụng HUBER LOSS thay vì MSE
        # Huber loss ít nhạy cảm với outliers hơn Mean Squared Error
        if self.use_huber_loss:
            # Smooth L1 Loss (Huber Loss)
            criterion = torch.nn.SmoothL1Loss()
        else:
            # Mean Squared Error Loss
            criterion = torch.nn.MSELoss()
        
        # Tính loss
        loss = criterion(state_action_values, expected_state_action_values)
        
        # Tối ưu hóa mô hình
        self.optimizer.zero_grad()
        loss.backward()
        
        # CẢI TIẾN 4: ÁP DỤNG GRADIENT CLIPPING
        # Giới hạn norm của gradient để tránh exploding gradients
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip_value)
        
        # Thực hiện bước tối ưu hóa
        self.optimizer.step()
        
        # Cập nhật target network theo chu kỳ, nếu đến thời điểm cập nhật
        self.update_target_network_if_needed()
        
        return loss.item()
            
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

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Lấy tham số của mô hình
        
        Returns:
            Từ điển chứa các tham số của mô hình
        """
        return {
            name: param.data.clone() 
            for name, param in self.policy_net.named_parameters()
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