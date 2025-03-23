import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Any, Optional

# Định nghĩa namedtuple Experience để lưu trữ kinh nghiệm trong replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class QNetwork(nn.Module):
    """
    Mạng neural cho Deep Q-Network.
    """
    def __init__(self, state_size: int, action_dim: List[int], hidden_sizes: List[int] = [128, 128]):
        """
        Khởi tạo mạng Q.
        
        Args:
            state_size: Kích thước không gian trạng thái
            action_dim: Danh sách các kích thước của không gian hành động
            hidden_sizes: Kích thước các lớp ẩn
        """
        super(QNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_dim = action_dim
        self.action_dim_product = np.prod(action_dim)
        
        # Lớp chung
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        
        # Lớp đầu ra riêng biệt cho từng chiều hành động
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[1], dim) for dim in action_dim
        ])
        
        # Lớp đầu ra cho hệ số Q chung
        self.value_layer = nn.Linear(hidden_sizes[1], 1)
    
    def forward(self, state: torch.Tensor) -> List[torch.Tensor]:
        """
        Lan truyền xuôi qua mạng.
        
        Args:
            state: Tensor trạng thái đầu vào
            
        Returns:
            List[torch.Tensor]: Danh sách các Q values cho từng chiều hành động
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Giá trị cơ sở cho trạng thái
        state_value = self.value_layer(x)
        
        # Q values cho mỗi chiều hành động
        action_values = [layer(x) for layer in self.output_layers]
        
        return action_values, state_value

class DQNAgent:
    """
    Agent sử dụng Deep Q-Network để học và ra quyết định trong môi trường blockchain.
    """
    def __init__(self, 
                 state_space: Dict[str, List],
                 action_space: Dict[str, List],
                 num_shards: int = 4,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 update_target_every: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Khởi tạo DQN Agent.
        
        Args:
            state_space: Dictionary mô tả không gian trạng thái
            action_space: Dictionary mô tả không gian hành động
            num_shards: Số lượng shard trong mạng
            learning_rate: Tốc độ học
            gamma: Hệ số giảm cho phần thưởng tương lai
            epsilon_start: Epsilon ban đầu cho chính sách ε-greedy
            epsilon_end: Epsilon tối thiểu
            epsilon_decay: Tốc độ giảm của epsilon
            buffer_size: Kích thước của replay buffer
            batch_size: Kích thước batch cho học tập
            update_target_every: Số bước để cập nhật mạng target
            device: Thiết bị để huấn luyện (CPU hoặc GPU)
        """
        self.state_space = state_space
        self.action_space = action_space
        self.num_shards = num_shards
        
        # Tính toán kích thước không gian trạng thái và hành động
        # State: 
        # - 4 đặc trưng cho mỗi shard: tắc nghẽn, giá trị giao dịch, điểm tin cậy, tỷ lệ thành công
        # - 3 đặc trưng toàn cục: số giao dịch đang chờ, độ trễ trung bình, tỷ lệ giao dịch xuyên shard
        # - 3 phần tử cho tỷ lệ đồng thuận hiện tại
        self.state_size = num_shards * 4 + 3 + 3
        
        # Action: Lựa chọn shard đích (0 to num_shards-1) và giao thức đồng thuận (0 to 2)
        self.action_dim = [num_shards, 3]
        
        # Tham số học tập
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.device = device
        
        # Đếm số bước đã thực hiện
        self.t_step = 0
        
        # Khởi tạo mạng Q chính và mạng Q target
        hidden_sizes = [128, 128]
        self.qnetwork_local = QNetwork(self.state_size, self.action_dim, hidden_sizes).to(self.device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Chỉ số batch hiện tại
        self.batch_idx = 0
    
    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """
        Thực hiện một bước học tập từ kinh nghiệm.
        
        Args:
            state: Trạng thái hiện tại
            action: Hành động đã thực hiện
            reward: Phần thưởng nhận được
            next_state: Trạng thái tiếp theo
            done: Trạng thái kết thúc
        """
        # Lưu kinh nghiệm vào replay buffer
        self.memory.append(Experience(state, action, reward, next_state, done))
        
        # Tăng bước
        self.t_step += 1
        
        # Học từ kinh nghiệm nếu đủ mẫu và đến thời điểm học
        if len(self.memory) > self.batch_size and self.t_step % 4 == 0:
            # Lấy một batch ngẫu nhiên từ memory
            experiences = random.sample(self.memory, self.batch_size)
            self._learn(experiences)
        
        # Cập nhật mạng target nếu đến thời điểm cập nhật
        if self.t_step % self.update_target_every == 0:
            self._update_target()
    
    def act(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Chọn hành động dựa trên trạng thái hiện tại.
        
        Args:
            state: Trạng thái hiện tại
            eval_mode: Nếu True, sẽ sử dụng chính sách tham lam (greedy policy)
            
        Returns:
            np.ndarray: Hành động được chọn [shard_id, consensus_protocol]
        """
        # Chuyển đổi state thành tensor và đưa vào device
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        # Set mạng Q sang chế độ đánh giá
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            # Lấy Q-values từ mạng
            action_values, _ = self.qnetwork_local(state)
            
        # Set mạng Q sang chế độ huấn luyện
        self.qnetwork_local.train()
        
        # Chính sách ε-greedy
        if eval_mode or random.random() > self.epsilon:
            # Chọn hành động tham lam
            actions = [torch.argmax(values).item() for values in action_values]
        else:
            # Chọn hành động ngẫu nhiên
            actions = [
                random.randrange(self.action_dim[0]),
                random.randrange(self.action_dim[1])
            ]
        
        return np.array(actions)
    
    def _learn(self, experiences: List[Experience]):
        """
        Cập nhật giá trị tham số dựa trên một batch của kinh nghiệm.
        
        Args:
            experiences: Danh sách các kinh nghiệm từ replay buffer
        """
        # Tạo các batch từ experiences
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(self.device)
        
        # Tính Q-value cho trạng thái hiện tại
        q_values_local, state_value_local = self.qnetwork_local(states)
        
        # Tính Q-value cho trạng thái tiếp theo từ mạng target
        q_values_target, state_value_target = self.qnetwork_target(next_states)
        
        # Tính max Q-value cho next state
        max_q_values = [values.max(1)[0].unsqueeze(1) for values in q_values_target]
        avg_max_q = sum(max_q_values) / len(max_q_values)
        
        # Tính target Q-value
        q_target = rewards + (self.gamma * avg_max_q * (1 - dones))
        
        # Tính loss cho mỗi chiều hành động
        losses = []
        for i, (action_values, action_dim_values) in enumerate(zip(q_values_local, actions.t())):
            action_idx = action_dim_values.unsqueeze(1)
            current_q = action_values.gather(1, action_idx)
            
            # Mean Squared Error loss
            loss_i = F.mse_loss(current_q, q_target.detach())
            losses.append(loss_i)
        
        # Tính tổng loss
        total_loss = sum(losses) / len(losses)
        
        # Cập nhật tham số
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping để ổn định quá trình học
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Cập nhật epsilon
        self._update_epsilon()
    
    def _update_target(self):
        """Cập nhật tham số từ mạng chính sang mạng target."""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
    
    def _update_epsilon(self):
        """Cập nhật epsilon theo lịch giảm dần."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        """
        Cập nhật mạng target với trọng số từ mạng local.
        Phương thức public gọi lại _update_target.
        """
        self._update_target()
        print(f"Đã cập nhật mạng target tại bước {self.t_step}")
    
    def train(self, env, num_episodes: int = 1000, max_steps: int = 1000, 
              print_every: int = 10, save_path: Optional[str] = None):
        """
        Huấn luyện agent trong môi trường.
        
        Args:
            env: Môi trường mô phỏng
            num_episodes: Số lượng episode
            max_steps: Số bước tối đa trong mỗi episode
            print_every: In kết quả sau mỗi số episode
            save_path: Đường dẫn để lưu mô hình
        """
        # Danh sách lưu phần thưởng mỗi episode
        scores = []
        
        # Danh sách lưu phần thưởng trung bình trong cửa sổ
        scores_window = deque(maxlen=100)
        
        for i_episode in range(1, num_episodes + 1):
            # Reset môi trường
            state = env.reset()
            score = 0
            
            for t in range(max_steps):
                # Chọn hành động từ trạng thái hiện tại
                action = self.act(state)
                
                # Thực hiện hành động và nhận phản hồi từ môi trường
                next_state, reward, done, _ = env.step(action)
                
                # Học từ kinh nghiệm
                self.step(state, action, reward, next_state, done)
                
                # Cập nhật trạng thái và điểm
                state = next_state
                score += reward
                
                if done:
                    break
            
            # Lưu điểm của episode
            scores.append(score)
            scores_window.append(score)
            
            # In thông tin tiến độ
            if i_episode % print_every == 0:
                print(f'Episode {i_episode}/{num_episodes} | Average Score: {np.mean(scores_window):.2f} | Epsilon: {self.epsilon:.2f}')
            
            # Lưu mô hình
            if save_path and i_episode % 100 == 0:
                torch.save(self.qnetwork_local.state_dict(), f'{save_path}/dqn_checkpoint_{i_episode}.pth')
        
        # Lưu mô hình cuối cùng
        if save_path:
            torch.save(self.qnetwork_local.state_dict(), f'{save_path}/dqn_final.pth')
        
        return scores
    
    def save(self, path: str):
        """
        Lưu mô hình.
        
        Args:
            path: Đường dẫn để lưu mô hình
        """
        torch.save({
            'qnetwork_local': self.qnetwork_local.state_dict(),
            'qnetwork_target': self.qnetwork_target.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """
        Tải mô hình từ file.
        
        Args:
            path: Đường dẫn đến file mô hình
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.qnetwork_local.load_state_dict(checkpoint['qnetwork_local'])
        self.qnetwork_target.load_state_dict(checkpoint['qnetwork_target'])
        self.epsilon = checkpoint['epsilon'] 