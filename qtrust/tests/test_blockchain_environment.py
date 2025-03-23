"""
Bài kiểm thử cho môi trường blockchain.
"""

import unittest
import numpy as np
import pytest
import gym

from qtrust.simulation.blockchain_environment import BlockchainEnvironment

class TestBlockchainEnvironment(unittest.TestCase):
    """
    Kiểm thử cho BlockchainEnvironment.
    """
    
    def setUp(self):
        """
        Thiết lập trước mỗi bài kiểm thử.
        """
        self.env = BlockchainEnvironment(
            num_shards=3,
            num_nodes_per_shard=5,
            max_transactions_per_step=50,
            transaction_value_range=(0.1, 50.0),
            max_steps=500
        )
        
    def test_initialization(self):
        """
        Kiểm thử khởi tạo môi trường.
        """
        # Kiểm tra số lượng shard và node
        self.assertEqual(self.env.num_shards, 3)
        self.assertEqual(self.env.num_nodes_per_shard, 5)
        self.assertEqual(self.env.total_nodes, 15)
        
        # Kiểm tra các giá trị khởi tạo khác
        self.assertEqual(self.env.max_transactions_per_step, 50)
        self.assertEqual(self.env.transaction_value_range, (0.1, 50.0))
        self.assertEqual(self.env.max_steps, 500)
        
        # Kiểm tra blockchain network đã được khởi tạo
        self.assertIsNotNone(self.env.blockchain_network)
        self.assertEqual(len(self.env.blockchain_network.nodes), 15)
        
        # Kiểm tra các tham số hình phạt và phần thưởng
        self.assertGreater(self.env.latency_penalty, 0)
        self.assertGreater(self.env.energy_penalty, 0)
        self.assertGreater(self.env.throughput_reward, 0)
        self.assertGreater(self.env.security_reward, 0)
        
    def test_reset(self):
        """
        Kiểm thử reset môi trường.
        """
        initial_state = self.env.reset()
        
        # Kiểm tra trạng thái đầu tiên
        self.assertIsNotNone(initial_state)
        self.assertEqual(self.env.current_step, 0)
        
        # Kiểm tra transaction pool đã được xóa
        self.assertEqual(len(self.env.transaction_pool), 0)
        
        # Kiểm tra metrics đã được reset
        self.assertEqual(self.env.performance_metrics['transactions_processed'], 0)
        self.assertEqual(self.env.performance_metrics['total_latency'], 0)
        self.assertEqual(self.env.performance_metrics['total_energy'], 0)
        
    def test_step(self):
        """
        Kiểm thử một bước trong môi trường.
        """
        _ = self.env.reset()
        
        # Tạo một action ngẫu nhiên hợp lệ
        action = self.env.action_space.sample()
        
        # Thực hiện một bước
        next_state, reward, done, info = self.env.step(action)
        
        # Kiểm tra state tiếp theo
        self.assertIsNotNone(next_state)
        
        # Kiểm tra reward
        self.assertIsInstance(reward, float)
        
        # Kiểm tra các thông tin
        self.assertIn('transactions_processed', info)
        self.assertIn('avg_latency', info)
        self.assertIn('avg_energy', info)
        self.assertIn('throughput', info)
        
        # Kiểm tra current step đã được tăng
        self.assertEqual(self.env.current_step, 1)
        
        # Kiểm tra done flag (nên là False vì mới bước đầu tiên)
        self.assertFalse(done)
        
    def test_multiple_steps(self):
        """
        Kiểm thử nhiều bước liên tiếp.
        """
        _ = self.env.reset()
        
        # Chạy 10 bước
        rewards = []
        for _ in range(10):
            action = self.env.action_space.sample()
            _, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            
            if done:
                break
        
        # Kiểm tra current step
        self.assertEqual(self.env.current_step, 10)
        
        # Kiểm tra số lượng phần thưởng
        self.assertEqual(len(rewards), 10)
        
    def test_generate_transactions(self):
        """
        Kiểm thử việc tạo giao dịch.
        """
        _ = self.env.reset()
        
        # Tạo giao dịch mới
        transactions = self.env._generate_transactions()
        
        # Kiểm tra số lượng giao dịch tạo ra
        self.assertLessEqual(len(transactions), self.env.max_transactions_per_step)
        
        # Kiểm tra định dạng giao dịch
        if transactions:
            tx = transactions[0]
            self.assertIn('source', tx)
            self.assertIn('destination', tx)
            self.assertIn('value', tx)
            self.assertIn('timestamp', tx)
            
            # Kiểm tra giá trị trong phạm vi
            self.assertGreaterEqual(tx['value'], self.env.transaction_value_range[0])
            self.assertLessEqual(tx['value'], self.env.transaction_value_range[1])
            
    def test_get_state(self):
        """
        Kiểm thử lấy trạng thái.
        """
        _ = self.env.reset()
        
        # Lấy trạng thái
        state = self.env._get_state()
        
        # Kiểm tra state không None
        self.assertIsNotNone(state)
        
        # Kiểm tra kích thước state
        self.assertIsInstance(state, np.ndarray)
        
    def test_calculate_reward(self):
        """
        Kiểm thử tính toán phần thưởng.
        """
        _ = self.env.reset()
        
        # Thêm một vài metrics giả lập
        self.env.performance_metrics['transactions_processed'] = 20
        self.env.performance_metrics['total_latency'] = 500
        self.env.performance_metrics['total_energy'] = 300
        
        # Tính toán phần thưởng
        reward, _ = self.env._calculate_reward()
        
        # Kiểm tra phần thưởng không None
        self.assertIsNotNone(reward)
        self.assertIsInstance(reward, float)
        
    def test_done_condition(self):
        """
        Kiểm thử điều kiện kết thúc.
        """
        _ = self.env.reset()
        
        # Chưa đạt max steps -> không done
        self.assertFalse(self.env._is_done())
        
        # Set current_step đến max_steps - 1
        self.env.current_step = self.env.max_steps - 1
        self.assertFalse(self.env._is_done())
        
        # Set current_step đến max_steps
        self.env.current_step = self.env.max_steps
        self.assertTrue(self.env._is_done())
        
    def test_action_space(self):
        """
        Kiểm thử không gian hành động.
        """
        # Kiểm tra không gian hành động
        self.assertIsInstance(self.env.action_space, gym.spaces.Space)
        
        # Lấy một hành động ngẫu nhiên
        action = self.env.action_space.sample()
        self.assertIsNotNone(action)
        
    def test_observation_space(self):
        """
        Kiểm thử không gian quan sát.
        """
        # Kiểm tra không gian quan sát
        self.assertIsInstance(self.env.observation_space, gym.spaces.Space)
        
        # Lấy một trạng thái
        state = self.env.reset()
        
        # Kiểm tra trạng thái nằm trong không gian quan sát
        self.assertTrue(self.env.observation_space.contains(state))
        
if __name__ == '__main__':
    unittest.main() 