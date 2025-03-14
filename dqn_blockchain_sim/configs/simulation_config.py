"""
Cấu hình mô phỏng Blockchain-DQN
"""

# Cấu hình mạng blockchain
BLOCKCHAIN_CONFIG = {
    # Thông số cơ bản
    "num_nodes": 100,               # Số lượng nút tổng cộng
    "num_shards": 8,                # Số lượng mảnh (shards)
    "min_nodes_per_shard": 5,       # Số lượng nút tối thiểu mỗi mảnh
    "block_time": 10,               # Thời gian tạo khối (giây)
    "transaction_timeout": 60,      # Thời gian timeout của giao dịch (giây)
    
    # Cấu hình bảo mật
    "security_level": "high",       # Mức độ bảo mật (low, medium, high)
    "encryption_algorithm": "aes-256-gcm",   # Thuật toán mã hóa
    "verification_percentage": 0.33,  # Tỷ lệ nút xác nhận giao dịch
    
    # Cấu hình tải hệ thống
    "transaction_rate": 1000,       # Số giao dịch trung bình mỗi giây
    "transaction_size": 2048,       # Kích thước giao dịch trung bình (bytes)
    "transaction_distribution": "normal",  # Phân phối lưu lượng giao dịch
    
    # Cấu hình tái phân mảnh
    "reshard_threshold": 0.75,      # Ngưỡng tải để kích hoạt tái phân mảnh
    "reshard_cooldown": 300,        # Thời gian chờ giữa các lần tái phân mảnh
    "max_reshard_nodes": 20,        # Số lượng nút tối đa được di chuyển mỗi lần
}

# Cấu hình DQN
DQN_CONFIG = {
    # Kiến trúc mạng
    "local_network_architecture": [128, 256, 128],  # Các lớp ẩn cho mạng cục bộ
    "coordination_network_architecture": [256, 512, 256],  # Các lớp ẩn cho mạng phối hợp
    
    # Thông số học
    "learning_rate": 1e-4,          # Tốc độ học
    "gamma": 0.99,                  # Hệ số chiết khấu
    "epsilon_start": 1.0,           # Giá trị epsilon khởi đầu (exploration)
    "epsilon_end": 0.1,             # Giá trị epsilon cuối
    "epsilon_decay": 0.0001,        # Tốc độ giảm epsilon
    
    # Cấu hình bộ nhớ
    "replay_buffer_size": 100000,   # Kích thước bộ nhớ lưu lại trải nghiệm
    "batch_size": 64,               # Kích thước batch để huấn luyện
    "target_update_freq": 1000,     # Tần suất cập nhật mạng mục tiêu
    
    # Cấu hình phần thưởng
    "reward_weights": {
        "throughput": 0.4,          # Trọng số cho thông lượng
        "latency": 0.3,             # Trọng số cho độ trễ
        "security": 0.2,            # Trọng số cho bảo mật
        "energy": 0.1,              # Trọng số cho tiêu thụ năng lượng
    },
}

# Cấu hình Federated Learning
FL_CONFIG = {
    # Cấu hình huấn luyện
    "rounds": 100,                  # Số vòng học liên kết
    "local_epochs": 5,              # Số epoch cục bộ mỗi vòng
    "min_clients": 5,               # Số lượng khách hàng tối thiểu mỗi vòng
    "client_fraction": 0.2,         # Tỷ lệ khách hàng tham gia mỗi vòng
    
    # Thông số bảo mật
    "dp_epsilon": 3.0,              # Tham số epsilon cho tính riêng tư vi phân
    "dp_delta": 1e-5,               # Tham số delta cho tính riêng tư vi phân
    "dp_clip_norm": 1.0,            # Giới hạn norm gradient cho DP
    
    # Cấu hình tổng hợp
    "aggregation_method": "fedavg_secure",  # Phương pháp tổng hợp
    "aggregation_frequency": 5,     # Tần suất tổng hợp mô hình
}

# Cấu hình TDCM
TDCM_CONFIG = {
    # Cấu hình tính điểm tin cậy
    "trust_factors": {
        "uptime": 0.2,
        "performance": 0.3,
        "validation_accuracy": 0.4,
        "reputation": 0.1
    },
    "performance_factors": {
        "response_time": 0.5,
        "throughput": 0.5
    },
    "validation_threshold": 0.7,
    "trust_threshold": 0.6,
    "reputation_decay": 0.95
}

# Cấu hình mô phỏng
SIMULATION_CONFIG = {
    "duration": 3600,               # Thời gian mô phỏng (giây)
    "warmup_time": 300,             # Thời gian khởi động (giây)
    "random_seed": 42,              # Seed cho tính ngẫu nhiên
    "log_level": "INFO",            # Mức độ ghi log
    "save_results": True,           # Lưu kết quả
    "visualization": True,          # Hiển thị kết quả trực quan
    "attack_scenarios": [           # Các kịch bản tấn công mô phỏng
        {
            "type": "sybil",        # Loại tấn công
            "start_time": 1000,     # Thời điểm bắt đầu (giây)
            "duration": 300,        # Thời gian (giây)
            "intensity": 0.1,       # Cường độ tấn công (0-1)
        },
        {
            "type": "ddos",
            "start_time": 2000,
            "duration": 200,
            "intensity": 0.2,
        },
    ],
    "num_shards": 8,
    "block_time": 10,  # Giây
    "network_latency": 100,  # ms
    "consensus_algorithm": "PoW",
    "steps": 500,
    "tx_per_step": 20,
    "shard_capacity": 100,
    "log_dir": "logs",
    "data_dir": "data",
    "visualize": True,
    "save_stats": True,
    "use_real_data": False,
    "use_dqn": True
}

# Cấu hình MAD-RAPID
MAD_RAPID_CONFIG = {
    "predictor_input_size": 8,
    "predictor_hidden_size": 128,
    "predictor_num_layers": 2,
    "optimizer_input_size": 16,
    "optimizer_hidden_size": 128,
    "use_dqn": True,
    "dqn_state_size": 12,
    "dqn_action_size": 3,
    "dqn_batch_size": 64,
    "dqn_gamma": 0.99,
    "dqn_epsilon": 1.0,
    "dqn_epsilon_min": 0.1,
    "dqn_epsilon_decay": 0.995,
    "dqn_learning_rate": 0.001,
    "dqn_tau": 0.01,
    "use_double_dqn": True,
    "use_dueling_dqn": False,
    "use_prioritized_replay": True,
    
    # Cấu hình Federated Learning
    "use_federated_learning": True,
    "fl_rounds": 5,
    "fl_local_epochs": 2,
    "fl_client_fraction": 0.8,
    "fl_aggregation_method": "fedavg",
    "fl_secure_aggregation": True
}

# Cấu hình ACSC
ACSC_CONFIG = {
    "default_strategy": "StandardPBFTConsensus",
    "strategies": {
        "FastBFTConsensus": {
            "min_trust_score": 0.8,
            "min_validators": 3,
            "max_validators": 7
        },
        "StandardPBFTConsensus": {
            "min_trust_score": 0.6,
            "min_validators": 5,
            "max_validators": 11
        },
        "RobustBFTConsensus": {
            "min_trust_score": 0.4,
            "min_validators": 7,
            "max_validators": 15
        }
    },
    "strategy_selection_window": 10,
    "performance_weight": 0.6,
    "security_weight": 0.4
}

# Cấu hình Federated Learning
FEDERATED_LEARNING_CONFIG = {
    "global_rounds": 5,
    "local_epochs": 2,
    "min_clients": 2,
    "client_fraction": 0.8,
    "aggregation_method": "fedavg",  # Một trong: "fedavg", "fedprox", "fedadam"
    "secure_aggregation": True,
    "privacy_budget": 1.0,  # Ngân sách điều chỉnh cho differential privacy
    "model_saving_interval": 5  # Lưu mô hình sau mỗi bao nhiêu vòng
} 