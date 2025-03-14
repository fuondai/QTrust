# DQN Blockchain Simulation

<p align="center">
  <img src="https://via.placeholder.com/150?text=DQN+Blockchain" alt="DQN Blockchain Logo"/>
</p>

> Mô phỏng blockchain tối ưu hóa bằng Deep Q-Network với các cơ chế đồng thuận thích ứng và hệ thống phân tích hiệu suất.

## Tổng quan

DQN Blockchain Simulation là một nền tảng nghiên cứu tiên tiến tích hợp học tăng cường dựa trên Deep Q-Network (DQN) vào hệ thống blockchain phân mảnh (sharded). Dự án phát triển các thuật toán và cơ chế mới để cải thiện hiệu suất, khả năng mở rộng và tính bảo mật của hệ thống blockchain.

### Các tính năng chính

- 🧠 **Deep Q-Network (DQN)** tối ưu hóa việc định tuyến và xử lý giao dịch
- 🔄 **Adaptive Cross-Shard Consensus (ACSC)** tự động lựa chọn thuật toán đồng thuận phù hợp
- 🌐 **MAD-RAPID Protocol** (Multi-Agent Dynamic Routing and Adaptive Path Intelligence Distribution)
- 🔐 **Hierarchical Trust-based Data Center Mechanism (HTDCM)** cải thiện tính bảo mật
- 📊 **Hệ thống phân tích hiệu suất toàn diện** với tạo báo cáo tự động
- 🔍 **Công cụ phân tích và so sánh** các phương pháp đồng thuận khác nhau
- 🖥️ **Giao diện người dùng đồ họa** để chạy các phân tích

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 1.8+
- NetworkX
- Matplotlib
- Pandas
- Seaborn
- FPDF (để tạo báo cáo PDF)
- TKinter (cho giao diện đồ họa)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Kiểm tra cài đặt

Sau khi cài đặt các thư viện, bạn có thể kiểm tra xem tất cả các module được import đúng hay không:

```bash
python test_imports.py
```

## Cấu trúc dự án

```
dqn_blockchain_sim/
├── agents/                # Các tác tử học tăng cường
│   ├── dqn_agent.py       # DQN agent cơ bản
│   └── enhanced_dqn.py    # DQN agent nâng cao (Dueling, Noisy, Prioritized)
├── blockchain/            # Thành phần blockchain
│   ├── adaptive_consensus.py  # Cơ chế đồng thuận thích ứng
│   ├── mad_rapid.py       # Giao thức Multi-Agent Dynamic RAPID
│   ├── network.py         # Quản lý mạng blockchain
│   ├── shard.py           # Triển khai shard
│   ├── shard_manager.py   # Quản lý và phối hợp shard
│   └── transaction.py     # Định nghĩa và xử lý giao dịch
├── tdcm/                  # Trust-based Data Center Mechanism
│   ├── hierarchical_trust.py  # Cơ chế tin cậy phân cấp
│   └── trust_manager.py   # Quản lý và đánh giá tin cậy
├── simulation/            # Mô phỏng
│   ├── advanced_simulation.py  # Mô phỏng blockchain-DQN nâng cao
│   └── main.py            # Điểm khởi đầu mô phỏng
├── experiments/           # Thử nghiệm và đánh giá
│   ├── benchmark_runner.py     # Chạy các benchmark
│   ├── consensus_comparison.py # So sánh các phương pháp đồng thuận
│   ├── generate_report.py      # Tạo báo cáo PDF
│   └── performance_analysis.py # Phân tích hiệu suất chi tiết
├── utils/                 # Tiện ích
│   └── real_data_builder.py    # Xây dựng bộ dữ liệu từ Ethereum
├── configs/               # Cấu hình
│   └── simulation_config.py    # Cấu hình mô phỏng
├── run_analysis.py        # Script chạy phân tích từ terminal
├── run_analysis_gui.py    # Giao diện đồ họa cho phân tích
└── requirements.txt       # Các thư viện phụ thuộc
```

## Kiến trúc và thành phần

### Sơ đồ kiến trúc tổng thể

```
+-------------------+       +----------------------+
|                   |       |                      |
|    DQN Agents     | <---> |  Blockchain Network  |
|                   |       |                      |
+-------------------+       +----------------------+
         ^                            ^
         |                            |
         v                            v
+-------------------+       +----------------------+
|                   |       |                      |
|   ACSC & HTDCM    | <---> |     MAD-RAPID       |
|                   |       |                      |
+-------------------+       +----------------------+
         ^                            ^
         |                            |
         v                            v
+-------------------+       +----------------------+
|                   |       |                      |
|    Simulation     | <---> |      Analytics      |
|                   |       |                      |
+-------------------+       +----------------------+
```

### Các thành phần chính

#### 1. Blockchain Network

Hệ thống blockchain phân mảnh (sharded) với các cơ chế đồng thuận khác nhau:

- **Sharding**: Chia mạng thành các shard nhỏ để tăng khả năng mở rộng
- **Cross-Shard Transactions**: Xử lý giao dịch giữa các shard khác nhau
- **Consensus Mechanisms**: Hỗ trợ PoW, PoS, PBFT và ACSC

#### 2. DQN Agents

Các tác tử học tăng cường dùng để tối ưu hóa:

- **ShardDQNAgent**: Học cách tối ưu định tuyến giao dịch trong shard
- **Enhanced DQN**: Cải tiến với Dueling Networks, Noisy Networks và Prioritized Replay

#### 3. Cơ chế đồng thuận thích ứng (ACSC)

- Tự động lựa chọn thuật toán đồng thuận dựa trên:
  - Giá trị giao dịch
  - Mức độ tin cậy
  - Tải mạng hiện tại

- Ba chiến lược đồng thuận:
  - **FastBFTConsensus**: Cho giao dịch giá trị thấp và tin cậy cao
  - **StandardPBFTConsensus**: Cân bằng giữa tốc độ và bảo mật
  - **RobustBFTConsensus**: Cho giao dịch giá trị cao, đòi hỏi bảo mật cao

#### 4. MAD-RAPID Protocol

Giao thức định tuyến thông minh đa tác tử:

- **Congestion Prediction**: Dự đoán tắc nghẽn mạng dựa trên lịch sử
- **Adaptive Path Optimization**: Tối ưu hóa đường đi dựa trên độ trễ và thông lượng
- **Intelligence Distribution**: Phân phối kiến thức giữa các node để cải thiện định tuyến

#### 5. Hierarchical Trust-based Data Center Mechanism (HTDCM)

- **Trust Scoring**: Đánh giá độ tin cậy của node dựa trên hành vi
- **Graph Neural Network**: Phân tích mối quan hệ giữa các node để phát hiện hành vi bất thường
- **Hierarchical Classification**: Phân loại node theo mức độ tin cậy

#### 6. Hệ thống phân tích hiệu suất

- **Performance Analysis**: Phân tích chi tiết hiệu suất theo nhiều chỉ số
- **Consensus Comparison**: So sánh các phương pháp đồng thuận
- **Report Generation**: Tạo báo cáo PDF với biểu đồ và phân tích

#### 7. Giao diện người dùng

- Giao diện dòng lệnh và GUI để chạy các phân tích
- Theo dõi tiến trình phân tích thời gian thực
- Xem và xuất báo cáo

## Chạy dự án

### Sử dụng scripts tự động

Dự án cung cấp hai scripts để chạy tự động với đường dẫn Python đầy đủ:

#### Windows:

```bash
run_project.bat
```

#### Linux/Mac:

```bash
chmod +x run_project.sh
./run_project.sh
```

### Chạy trực tiếp

Nếu bạn muốn chạy trực tiếp, hãy sử dụng một trong các lệnh sau với đường dẫn Python đầy đủ:

```bash
# Đường dẫn đầy đủ đến Python
PYTHON_PATH=$(which python3 2>/dev/null || which python)

# Chạy mô phỏng cơ bản
$PYTHON_PATH -m dqn_blockchain_sim.simulation.main

# Chạy mô phỏng nâng cao
$PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5

# Chạy với visualize
$PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --visualize

# Chạy và lưu kết quả
$PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --save_stats

# Tạo báo cáo phân tích
$PYTHON_PATH -m dqn_blockchain_sim.experiments.generate_report

# Chạy giao diện phân tích
$PYTHON_PATH -m dqn_blockchain_sim.run_analysis_gui
```

### Chạy và ghi log

Bạn cũng có thể sử dụng script run_and_log.py để chạy mô phỏng và ghi log:

```bash
python run_and_log.py
```

## Luồng dữ liệu

1. **Khởi tạo mô phỏng**: Tạo mạng blockchain với các shard và cấu hình DQN
2. **Tạo giao dịch**: Tạo các giao dịch trong/giữa các shard
3. **Tối ưu hóa DQN**: DQN agent chọn cách tốt nhất để định tuyến giao dịch
4. **Xử lý đồng thuận**: ACSC chọn thuật toán đồng thuận phù hợp
5. **Thu thập dữ liệu**: Thu thập số liệu về hiệu suất, độ trễ, tiêu thụ năng lượng
6. **Phân tích kết quả**: Phân tích kết quả và tạo báo cáo

## Hướng dẫn mở rộng

### Thêm thuật toán DQN mới

Để thêm một thuật toán DQN mới, hãy tạo một lớp kế thừa từ `EnhancedDQNAgent` trong `agents/enhanced_dqn.py`:

```python
class YourNewDQNAgent(EnhancedDQNAgent):
    def __init__(self, state_dim, action_dim, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        # Thêm các thuộc tính mới

    def select_action(self, state):
        # Triển khai lựa chọn hành động mới
        pass
        
    def update(self, state, action, reward, next_state, done):
        # Triển khai cập nhật mới
        pass
```

### Thêm phương pháp đồng thuận mới

Để thêm một phương pháp đồng thuận mới, hãy tạo một lớp kế thừa từ `ConsensusStrategy` trong `blockchain/adaptive_consensus.py`:

```python
class YourNewConsensus(ConsensusStrategy):
    def __init__(self):
        super().__init__()
        # Khởi tạo
        
    def verify_transaction(self, transaction):
        # Triển khai xác minh giao dịch
        pass
        
    def calculate_energy_consumption(self, num_validators, transaction_value):
        # Tính toán tiêu thụ năng lượng
        pass
```

### Thêm chỉ số phân tích mới

Để thêm một chỉ số phân tích mới, sửa đổi lớp `PerformanceAnalyzer` trong `experiments/performance_analysis.py`:

```python
def analyze_your_new_metric(self) -> pd.DataFrame:
    """
    Phân tích chỉ số mới
    
    Returns:
        DataFrame kết quả phân tích
    """
    # Triển khai phân tích
    pass
```

## Đóng góp

Chúng tôi rất hoan nghênh đóng góp! Hãy tạo Pull Request hoặc báo lỗi qua Issues.

## Tài liệu tham khảo

1. Vitalik Buterin et al., "Ethereum 2.0: A Complete Guide", Ethereum Foundation, 2020
2. Volodymyr Mnih et al., "Human-level control through deep reinforcement learning", Nature, 2015
3. Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML, 2016
4. T. Schaul et al., "Prioritized Experience Replay", ICLR, 2016

## Giấy phép

MIT License 