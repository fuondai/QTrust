#!/usr/bin/env python3
"""
Script để chạy mô phỏng blockchain tối ưu hóa dựa trên DQN
"""

import argparse
import sys
import logging
import time
from dqn_blockchain_sim.simulation.main import Simulation
from dqn_blockchain_sim.configs.simulation_config import SIMULATION_CONFIG


def setup_logger():
    """Thiết lập logger"""
    logger = logging.getLogger('blockchain_sim')
    logger.setLevel(logging.INFO)
    
    # Tạo handler cho console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Thêm handler vào logger
    logger.addHandler(console_handler)
    
    return logger


def parse_arguments():
    """Phân tích tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Chạy mô phỏng blockchain tối ưu hóa bằng DQN')
    
    parser.add_argument('--duration', type=int, default=None,
                        help='Thời gian mô phỏng (giây)')
    
    parser.add_argument('--nodes', type=int, default=None,
                        help='Số lượng nút trong mạng')
    
    parser.add_argument('--shards', type=int, default=None,
                        help='Số lượng shard')
    
    parser.add_argument('--tx-rate', type=int, default=None,
                        help='Tỷ lệ giao dịch trên giây')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Hiển thị kết quả sau khi mô phỏng')
    
    parser.add_argument('--save', action='store_true',
                        help='Lưu kết quả mô phỏng')
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed cho tính ngẫu nhiên')
    
    parser.add_argument('--verbose', action='store_true',
                        help='In thông tin chi tiết trong quá trình mô phỏng')
    
    return parser.parse_args()


def main():
    """Hàm chính"""
    logger = setup_logger()
    args = parse_arguments()
    
    # Tạo cấu hình từ tham số dòng lệnh
    config = SIMULATION_CONFIG.copy()
    
    if args.duration is not None:
        config["duration"] = args.duration
        
    if args.seed is not None:
        config["random_seed"] = args.seed
        
    if args.visualize:
        config["visualization"] = True
        
    if args.save:
        config["save_results"] = True
        
    # Tạo cấu hình blockchain nếu cần
    blockchain_config = {}
    
    if args.nodes is not None:
        blockchain_config["num_nodes"] = args.nodes
        
    if args.shards is not None:
        blockchain_config["num_shards"] = args.shards
        
    if args.tx_rate is not None:
        blockchain_config["transaction_rate"] = args.tx_rate
    
    # Đặt cấp độ log
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        config["log_level"] = "DEBUG"
    
    # In thông tin mô phỏng
    logger.info("Bắt đầu mô phỏng blockchain tối ưu hóa dựa trên DQN")
    logger.info(f"Thời gian mô phỏng: {config['duration']} giây")
    
    # Đo thời gian thực thi
    start_time = time.time()
    
    # Tạo và chạy mô phỏng
    simulation = Simulation(config)
    simulation.run_simulation()
    
    # In thông tin thời gian thực thi
    elapsed_time = time.time() - start_time
    logger.info(f"Mô phỏng hoàn tất trong {elapsed_time:.2f} giây")


if __name__ == "__main__":
    main() 