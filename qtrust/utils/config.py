"""
Module chứa các hàm quản lý cấu hình cho hệ thống QTrust.
"""

import os
import yaml
import json
import argparse
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

class QTrustConfig:
    """
    Lớp quản lý cấu hình cho hệ thống QTrust.
    """
    
    DEFAULT_CONFIG = {
        # Thông số môi trường
        "environment": {
            "num_shards": 4,
            "num_nodes_per_shard": 10,
            "max_transactions_per_step": 100,
            "transaction_value_range": [0.1, 100.0],
            "max_steps": 1000,
            "latency_penalty": 0.1,
            "energy_penalty": 0.1,
            "throughput_reward": 1.0,
            "security_reward": 1.0,
            "cross_shard_reward": 0.5,
            "seed": 42
        },
        
        # Thông số DQN Agent
        "dqn_agent": {
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 0.995,
            "buffer_size": 10000,
            "batch_size": 64,
            "target_update": 10,
            "hidden_dim": 128,
            "num_episodes": 500
        },
        
        # Thông số Adaptive Consensus
        "consensus": {
            "transaction_threshold_low": 10.0,
            "transaction_threshold_high": 50.0,
            "congestion_threshold": 0.7,
            "min_trust_threshold": 0.3,
            "fastbft_latency": 0.2,
            "fastbft_energy": 0.2,
            "fastbft_security": 0.5,
            "pbft_latency": 0.5,
            "pbft_energy": 0.5,
            "pbft_security": 0.8,
            "robustbft_latency": 0.8,
            "robustbft_energy": 0.8,
            "robustbft_security": 0.95
        },
        
        # Thông số MAD-RAPID Router
        "routing": {
            "congestion_weight": 1.0,
            "latency_weight": 0.7,
            "energy_weight": 0.5,
            "trust_weight": 0.8,
            "prediction_horizon": 5,
            "congestion_threshold": 0.8,
            "weight_adjustment_rate": 0.1
        },
        
        # Thông số HTDCM
        "trust": {
            "local_update_weight": 0.7,
            "global_update_weight": 0.3,
            "initial_trust": 0.5,
            "trust_threshold": 0.3,
            "penalty_factor": 0.2,
            "reward_factor": 0.1,
            "observation_window": 50,
            "suspicious_threshold": 0.7,
            "malicious_threshold": 0.9
        },
        
        # Thông số Federated Learning
        "federated": {
            "num_rounds": 20,
            "local_epochs": 5,
            "fraction_fit": 0.8,
            "min_fit_clients": 3,
            "min_available_clients": 4,
            "batch_size": 32,
            "learning_rate": 0.01,
            "trust_threshold": 0.4
        },
        
        # Thông số về visualization
        "visualization": {
            "save_plots": True,
            "plot_frequency": 50,
            "output_dir": "results"
        },
        
        # Thông số trình mô phỏng
        "simulation": {
            "num_transactions": 1000,
            "cross_shard_prob": 0.3,
            "honest_node_prob": 0.9,
            "num_network_events": 20,
            "num_malicious_activities": 10
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Khởi tạo đối tượng cấu hình.
        
        Args:
            config_path: Đường dẫn đến file cấu hình. Nếu là None, 
                        sử dụng cấu hình mặc định.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is not None:
            self.load_config(config_path)
            
        # Đảm bảo thư mục output tồn tại
        output_dir = self.config["visualization"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
            
    def load_config(self, config_path: str) -> None:
        """
        Tải cấu hình từ file.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        extension = os.path.splitext(config_path)[1].lower()
        
        if not os.path.exists(config_path):
            print(f"Cảnh báo: File cấu hình {config_path} không tồn tại. Sử dụng cấu hình mặc định.")
            return
        
        try:
            if extension == '.yaml' or extension == '.yml':
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif extension == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {extension}")
                
            # Cập nhật cấu hình với dữ liệu từ file
            self._update_nested_dict(self.config, config_data)
            
        except Exception as e:
            print(f"Lỗi khi tải file cấu hình: {e}")
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cập nhật dictionary một cách đệ quy, giữ nguyên cấu trúc lồng nhau.
        
        Args:
            d: Dictionary đích
            u: Dictionary nguồn
            
        Returns:
            Dictionary đã được cập nhật
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def save_config(self, config_path: str) -> None:
        """
        Lưu cấu hình hiện tại vào file.
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        extension = os.path.splitext(config_path)[1].lower()
        
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            
            if extension == '.yaml' or extension == '.yml':
                with open(config_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
            elif extension == '.json':
                with open(config_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Định dạng file không được hỗ trợ: {extension}")
                
        except Exception as e:
            print(f"Lỗi khi lưu file cấu hình: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Lấy giá trị của một tham số cấu hình.
        
        Args:
            key: Khóa của tham số (có thể dùng dấu chấm cho nested keys)
            default: Giá trị mặc định nếu không tìm thấy khóa
            
        Returns:
            Giá trị của tham số hoặc giá trị mặc định
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        Đặt giá trị cho một tham số cấu hình.
        
        Args:
            key: Khóa của tham số (có thể dùng dấu chấm cho nested keys)
            value: Giá trị cần đặt
        """
        keys = key.split('.')
        current = self.config
        
        for i, k in enumerate(keys[:-1]):
            if isinstance(current, dict):
                if k not in current:
                    current[k] = {}
                current = current[k]
            else:
                raise ValueError(f"Không thể đặt giá trị cho {key}: {'.'.join(keys[:i+1])} không phải là dictionary")
                
        if isinstance(current, dict):
            current[keys[-1]] = value
        else:
            raise ValueError(f"Không thể đặt giá trị cho {key}: {'.'.join(keys[:-1])} không phải là dictionary")
    
    def get_all(self) -> Dict[str, Any]:
        """
        Lấy toàn bộ cấu hình.
        
        Returns:
            Dictionary chứa toàn bộ cấu hình
        """
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Lấy một phần cấu hình.
        
        Args:
            section: Tên của phần cấu hình
            
        Returns:
            Dictionary chứa phần cấu hình được yêu cầu
        """
        if section in self.config:
            return self.config[section].copy()
        return {}

def parse_arguments() -> argparse.Namespace:
    """
    Phân tích các đối số dòng lệnh.
    
    Returns:
        Namespace chứa các đối số dòng lệnh đã được phân tích
    """
    parser = argparse.ArgumentParser(description='QTrust - Hệ thống Blockchain thông minh với Deep Reinforcement Learning')
    
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Đường dẫn đến file cấu hình')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed ngẫu nhiên cho tính tái lập')
    
    parser.add_argument('--num-shards', type=int, default=None,
                        help='Số lượng shard trong mạng blockchain')
    
    parser.add_argument('--num-nodes-per-shard', type=int, default=None,
                        help='Số lượng node trong mỗi shard')
    
    parser.add_argument('--num-episodes', type=int, default=None,
                        help='Số lượng episode cho huấn luyện DQN')
    
    parser.add_argument('--num-transactions', type=int, default=None,
                        help='Số lượng giao dịch để mô phỏng')
    
    parser.add_argument('--cross-shard-prob', type=float, default=None,
                        help='Xác suất giao dịch xuyên shard')
    
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Tốc độ học cho DQN Agent')
    
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Thư mục đầu ra để lưu kết quả')
    
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'simulate'],
                        default='train', help='Chế độ chạy (train, evaluate, simulate)')
    
    parser.add_argument('--save-model', action='store_true',
                        help='Lưu mô hình sau khi huấn luyện')
    
    parser.add_argument('--load-model', type=str, default=None,
                        help='Tải mô hình từ đường dẫn đã cho')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Tạo vizualization trong quá trình chạy')
    
    return parser.parse_args()

def update_config_from_args(config: QTrustConfig, args: argparse.Namespace) -> None:
    """
    Cập nhật cấu hình từ các đối số dòng lệnh.
    
    Args:
        config: Đối tượng cấu hình để cập nhật
        args: Namespace chứa các đối số dòng lệnh
    """
    # Ánh xạ các đối số đến các khóa cấu hình
    arg_to_config = {
        'seed': 'environment.seed',
        'num_shards': 'environment.num_shards',
        'num_nodes_per_shard': 'environment.num_nodes_per_shard',
        'num_episodes': 'dqn_agent.num_episodes',
        'num_transactions': 'simulation.num_transactions',
        'cross_shard_prob': 'simulation.cross_shard_prob',
        'learning_rate': 'dqn_agent.learning_rate',
        'output_dir': 'visualization.output_dir'
    }
    
    # Cập nhật cấu hình nếu đối số được cung cấp
    for arg_name, config_key in arg_to_config.items():
        arg_value = getattr(args, arg_name)
        if arg_value is not None:
            config.set(config_key, arg_value)
            
    # Xử lý các đối số khác
    if args.visualize:
        config.set('visualization.save_plots', True)

def load_config_from_args() -> QTrustConfig:
    """
    Tải cấu hình từ các đối số dòng lệnh.
    
    Returns:
        Đối tượng cấu hình đã được cập nhật
    """
    args = parse_arguments()
    config = QTrustConfig(args.config)
    update_config_from_args(config, args)
    return config, args 