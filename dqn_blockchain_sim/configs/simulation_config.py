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
        "uptime": 0.25,             # Trọng số cho thời gian hoạt động
        "performance": 0.30,        # Trọng số cho hiệu suất
        "security_incidents": 0.25, # Trọng số cho các sự cố bảo mật
        "energy_efficiency": 0.20,  # Trọng số cho hiệu quả năng lượng
    },
    
    # Thông số cập nhật điểm tin cậy
    "trust_update_frequency": 10,   # Tần suất cập nhật điểm tin cậy (giây)
    "trust_decay_rate": 0.01,       # Tốc độ suy giảm điểm tin cậy
    "trust_history_weight": 0.7,    # Trọng số lịch sử cho điểm tin cậy
    
    # Thông số lựa chọn
    "selection_batch_size": 10,     # Số lượng nút được xem xét mỗi lần lựa chọn
    "min_trust_threshold": 0.5,     # Ngưỡng tin cậy tối thiểu để được chọn
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
} 