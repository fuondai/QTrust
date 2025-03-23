"""
Bài kiểm thử cho DQN agent.
"""

import unittest
import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete, MultiDiscrete

from qtrust.agents.dqn_agent import DQNAgent, QNetwork, Experience

class SimpleEnv(gym.Env):
    """
    Môi trường đơn giản cho kiểm thử.
    """
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        self.action_space = MultiDiscrete([3, 2])  # Hai không gian hành động rời rạc
        self.state = np.zeros(5, dtype=np.float32)
        self.step_count = 0
        
    def reset(self):
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        self.step_count = 0
        return self.state
    
    def step(self, action):
        self.step_count += 1
        self.state = np.random.uniform(-1, 1, size=5).astype(np.float32)
        reward = 1.0 if action[0] == 1 else -0.1
        done = self.step_count >= 10
        info = {}
        return self.state, reward, done, info

class TestQNetwork(unittest.TestCase):
    """
    Kiểm thử cho QNetwork.
    """
    
    def setUp(self):
        self.state_size = 5
        self.action_dim = [3, 2]  # Hai không gian hành động rời rạc
        self.hidden_sizes = [64, 64]
        self.network = QNetwork(self.state_size, self.action_dim, self.hidden_sizes)
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo mạng Q.
        """
        # Kiểm tra các tham số
        self.assertEqual(self.network.state_size, self.state_size)
        self.assertEqual(self.network.action_dim, self.action_dim)
        
        # Kiểm tra các lớp
        self.assertEqual(self.network.fc1.in_features, self.state_size)
        self.assertEqual(self.network.fc1.out_features, self.hidden_sizes[0])
        
        self.assertEqual(self.network.fc2.in_features, self.hidden_sizes[0])
        self.assertEqual(self.network.fc2.out_features, self.hidden_sizes[1])
        
        # Kiểm tra các lớp đầu ra
        self.assertEqual(len(self.network.output_layers), len(self.action_dim))
        self.assertEqual(self.network.output_layers[0].out_features, self.action_dim[0])
        self.assertEqual(self.network.output_layers[1].out_features, self.action_dim[1])
        
        # Kiểm tra lớp giá trị
        self.assertEqual(self.network.value_layer.out_features, 1)
        
    def test_forward_pass(self):
        """
        Kiểm thử pass thuận (forward pass).
        """
        # Tạo input ngẫu nhiên
        batch_size = 10
        x = torch.randn(batch_size, self.state_size)
        
        # Forward pass
        action_values, state_value = self.network(x)
        
        # Kiểm tra kích thước output
        self.assertEqual(len(action_values), len(self.action_dim))
        self.assertEqual(action_values[0].shape, (batch_size, self.action_dim[0]))
        self.assertEqual(action_values[1].shape, (batch_size, self.action_dim[1]))
        self.assertEqual(state_value.shape, (batch_size, 1))

class TestDQNAgent(unittest.TestCase):
    """
    Kiểm thử cho DQNAgent.
    """
    
    def setUp(self):
        self.env = SimpleEnv()
        
        # Định nghĩa không gian trạng thái và hành động
        self.state_space = {
            'network_congestion': [0.0, 1.0],  # Mức độ tắc nghẽn
            'transaction_value': [0.1, 100.0],
            'trust_scores': [0.0, 1.0],  # Điểm tin cậy
            'success_rate': [0.0, 1.0]   # Tỷ lệ thành công
        }
        
        self.action_space = {
            'shard_selection': list(range(3)),  # 3 shard
            'consensus_protocol': list(range(2))  # 2 protocol
        }
        
        self.agent = DQNAgent(
            state_space=self.state_space,
            action_space=self.action_space,
            num_shards=3,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            buffer_size=1000,
            batch_size=64,
            update_target_every=100
        )
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo agent.
        """
        # Kiểm tra các tham số
        self.assertEqual(self.agent.num_shards, 3)
        self.assertEqual(self.agent.gamma, 0.99)
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertEqual(self.agent.epsilon_end, 0.1)
        self.assertEqual(self.agent.epsilon_decay, 0.995)
        self.assertEqual(self.agent.batch_size, 64)
        
        # Kiểm tra không gian trạng thái và hành động
        self.assertEqual(self.agent.state_size, 3 * 4 + 3 + 3)  # 3 shards * 4 features + global features
        self.assertEqual(self.agent.action_dim, [3, 2])
        
        # Kiểm tra các network
        self.assertIsInstance(self.agent.qnetwork_local, QNetwork)
        self.assertIsInstance(self.agent.qnetwork_target, QNetwork)
        
        # Kiểm tra optimizer
        self.assertIsInstance(self.agent.optimizer, torch.optim.Adam)
        
        # Kiểm tra buffer trống ban đầu
        self.assertEqual(len(self.agent.memory), 0)
        
    def test_step(self):
        """
        Kiểm thử hàm step.
        """
        # Tạo vài experience và thêm vào buffer
        state = np.random.uniform(-1, 1, size=self.agent.state_size).astype(np.float32)
        action = np.array([1, 0])  # Hành động mẫu [shard_id, consensus_protocol]
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.agent.state_size).astype(np.float32)
        done = False
        
        # Thêm 100 experience
        for _ in range(100):
            self.agent.step(state, action, reward, next_state, done)
            
        # Kiểm tra buffer
        self.assertEqual(len(self.agent.memory), 100)
        
    def test_act(self):
        """
        Kiểm thử hàm act.
        """
        # Tạo state ngẫu nhiên
        state = np.random.uniform(-1, 1, size=self.agent.state_size).astype(np.float32)
        
        # Thực hiện hành động với chế độ đánh giá (eval_mode=True)
        action = self.agent.act(state, eval_mode=True)
        
        # Kiểm tra shape của hành động
        self.assertEqual(action.shape, (2,))
        self.assertIn(action[0], range(3))  # Shard ID 0-2
        self.assertIn(action[1], range(2))  # Consensus protocol 0-1
        
    def test_learn(self):
        """
        Kiểm thử quá trình học.
        """
        # Thêm đủ experience để học
        state = np.random.uniform(-1, 1, size=self.agent.state_size).astype(np.float32)
        action = np.array([1, 0])  # Hành động mẫu [shard_id, consensus_protocol]
        reward = 1.0
        next_state = np.random.uniform(-1, 1, size=self.agent.state_size).astype(np.float32)
        done = False
        
        # Lấy các tham số trước khi học
        params_before = [p.clone().detach() for p in self.agent.qnetwork_local.parameters()]
        
        # Thêm nhiều experience vào buffer
        for _ in range(self.agent.batch_size + 10):  # Thêm nhiều hơn batch_size
            self.agent.step(state, action, reward, next_state, done)
            
        # Các tham số đã thay đổi sau nhiều bước
        params_after = [p.clone().detach() for p in self.agent.qnetwork_local.parameters()]
        
        # Kiểm tra xem ít nhất một số tham số đã thay đổi (do học tập)
        any_param_changed = False
        for before, after in zip(params_before, params_after):
            diff = torch.sum(torch.abs(after - before))
            if diff.item() > 0:
                any_param_changed = True
                break
        
        self.assertTrue(any_param_changed)
            
    def test_update_target(self):
        """
        Kiểm thử hàm update target network.
        """
        # Thay đổi local network
        for param in self.agent.qnetwork_local.parameters():
            param.data = torch.randn_like(param.data)
            
        # Kiểm tra trước khi update, 2 networks khác nhau
        for local_param, target_param in zip(self.agent.qnetwork_local.parameters(), 
                                           self.agent.qnetwork_target.parameters()):
            diff = torch.sum(torch.abs(local_param - target_param))
            self.assertGreater(diff.item(), 0)
            
        # Update target network
        self.agent._update_target()
        
        # Kiểm tra sau khi update, 2 networks giống nhau
        for local_param, target_param in zip(self.agent.qnetwork_local.parameters(), 
                                           self.agent.qnetwork_target.parameters()):
            diff = torch.sum(torch.abs(local_param - target_param))
            self.assertEqual(diff.item(), 0)
            
    def test_epsilon_decay(self):
        """
        Kiểm thử epsilon decay.
        """
        # Lưu epsilon ban đầu
        initial_epsilon = self.agent.epsilon
        
        # Thực hiện epsilon decay
        for _ in range(10):
            self.agent._update_epsilon()
            
        # Kiểm tra epsilon đã giảm
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
        # Kiểm tra epsilon không thấp hơn epsilon_end
        for _ in range(1000):
            self.agent._update_epsilon()
            
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_end)
        
    def test_training_episode(self):
        """
        Kiểm thử đào tạo một episode.
        """
        # Đào tạo 1 episode
        env = SimpleEnv()
        total_reward = 0
        state = env.reset()
        
        for _ in range(10):
            action = self.agent.act(state)
            next_state, reward, done, _ = env.step(action)
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        # Không có kiểm tra cụ thể, chỉ đảm bảo không có lỗi khi chạy
        # Và episode hoàn thành thành công

if __name__ == '__main__':
    unittest.main() 