# DQN Blockchain Simulation

Hệ thống mô phỏng blockchain phân mảnh sử dụng học tăng cường (Deep Q-Learning) với nhiều cải tiến tiên tiến cho quy mô lớn và hiệu suất cao.

## Tổng quan

Dự án này mô phỏng một mạng blockchain phân mảnh (sharded blockchain) tích hợp nhiều cơ chế tiên tiến để cải thiện hiệu suất và khả năng mở rộng. Các cơ chế này bao gồm:

1. **MAD-RAPID Protocol** - Giao thức tối ưu hóa định tuyến đa tác tử cho giao dịch xuyên mảnh
2. **HTDCM** (Hierarchical Trust-based Data Center Mechanism) - Cơ chế tin cậy phân cấp dựa trên trung tâm dữ liệu
3. **ACSC** (Adaptive Cross-Shard Consensus) - Cơ chế đồng thuận thích ứng xuyên mảnh
4. **DQN Agent** - Tác tử học tăng cường để tối ưu hóa hiệu suất của shard

Hệ thống cũng hỗ trợ sử dụng dữ liệu thực từ mạng Ethereum để mô phỏng giao dịch thực tế.

## Cài đặt

### Yêu cầu

- Python 3.7+
- PyTorch
- NumPy, Matplotlib, NetworkX
- pandas, tqdm

### Cài đặt các gói phụ thuộc

```bash
pip install numpy matplotlib networkx torch pandas tqdm
```

### Cấu trúc thư mục

```
dqn_blockchain_sim/
├── agents/                # Các tác tử DQN
│   ├── __init__.py
│   └── dqn_agent.py
├── blockchain/            # Các thành phần blockchain
│   ├── __init__.py
│   ├── network.py
│   ├── transaction.py
│   ├── mad_rapid.py       # Giao thức MAD-RAPID
│   └── adaptive_consensus.py  # Cơ chế ACSC
├── tdcm/                  # Cơ chế HTDCM
│   ├── __init__.py
│   └── hierarchical_trust.py
├── utils/                 # Tiện ích
│   ├── __init__.py
│   └── real_data_builder.py
├── simulation/            # Mô-đun mô phỏng
│   ├── __init__.py
│   ├── main.py
│   ├── mad_rapid_integration.py
│   ├── htdcm_integration.py
│   ├── acsc_integration.py
│   ├── real_data_integration.py
│   └── advanced_simulation.py
├── run_simulation.py      # Script mô phỏng cơ bản
└── run_advanced_simulation.py  # Script mô phỏng nâng cao
```

## Sử dụng

### Chạy mô phỏng nâng cao

Mô phỏng nâng cao tích hợp tất cả các cải tiến tiên tiến và hỗ trợ nhiều tùy chọn:

```bash
python -m dqn_blockchain_sim.run_advanced_simulation --num_shards 8 --steps 500 --visualize --save_stats
```

### Các tham số

- `--num_shards`: Số lượng shard trong mạng (mặc định: 8)
- `--steps`: Số bước mô phỏng (mặc định: 500)
- `--tx_per_step`: Số giao dịch mỗi bước (mặc định: 20)
- `--use_real_data`: Sử dụng dữ liệu Ethereum thực
- `--eth_api_key`: Khóa API Ethereum (bắt buộc nếu sử dụng `--use_real_data`)
- `--use_dqn`: Sử dụng DQN agent (mặc định: bật)
- `--data_dir`: Thư mục lưu dữ liệu (mặc định: "data")
- `--log_dir`: Thư mục lưu nhật ký (mặc định: "logs")
- `--visualize`: Hiển thị đồ thị kết quả
- `--save_stats`: Lưu thống kê vào file

### Sử dụng dữ liệu thực từ Ethereum

Để chạy mô phỏng với dữ liệu thực từ Ethereum:

```bash
python -m dqn_blockchain_sim.run_advanced_simulation --use_real_data --eth_api_key YOUR_ETHEREUM_API_KEY
```

Lưu ý: Bạn cần có API key từ một nhà cung cấp dữ liệu Ethereum (ví dụ: Infura, Alchemy).

## Các cải tiến chính

### MAD-RAPID Protocol

Giao thức định tuyến đa tác tử động (Multi-Agent Dynamic RAPID) tối ưu hóa đường đi cho các giao dịch xuyên mảnh, giảm độ trễ và tiết kiệm năng lượng. Giao thức này sử dụng dự đoán tắc nghẽn và tính năng nén để cải thiện hiệu suất.

### HTDCM (Hierarchical Trust-based Data Center Mechanism)

Cơ chế độ tin cậy phân cấp sử dụng mạng nơ-ron đồ thị (GNN) để tính toán và duy trì điểm tin cậy cho các shard và trung tâm dữ liệu. Điểm tin cậy này được sử dụng để xác định mức độ bảo mật cần thiết cho các giao dịch.

### ACSC (Adaptive Cross-Shard Consensus)

Cơ chế đồng thuận thích ứng cho phép chọn chiến lược đồng thuận phù hợp (nhanh, tiêu chuẩn, hoặc mạnh mẽ) dựa trên giá trị giao dịch, mức độ bảo mật, và điểm tin cậy của các shard liên quan.

### DQN Agent

Các tác tử DQN trong mỗi shard học cách tối ưu hóa các quyết định xử lý giao dịch để cải thiện thông lượng, giảm độ trễ và giảm tắc nghẽn.

## Phân tích kết quả

Kết quả mô phỏng được hiển thị trực quan (nếu sử dụng `--visualize`) và bao gồm các thống kê chi tiết:

1. **Thông lượng**: Số giao dịch xử lý mỗi bước
2. **Độ trễ**: Thời gian trung bình để xử lý giao dịch
3. **Tỉ lệ thành công**: Tỉ lệ giao dịch được xử lý thành công
4. **Mức độ tắc nghẽn**: Đo lường mức độ tắc nghẽn trong mạng
5. **Tiêu thụ năng lượng**: Ước tính năng lượng sử dụng
6. **Thống kê của từng module**: Hiệu suất chi tiết của MAD-RAPID, HTDCM và ACSC

## Đóng góp

Dự án này có thể được mở rộng theo nhiều cách:

1. Thêm chiến lược học tăng cường mới
2. Cải thiện mô hình mạng blockchain
3. Tích hợp dữ liệu thực từ các mạng blockchain khác
4. Thêm các cơ chế đồng thuận và bảo mật mới

## Liên hệ

Nếu bạn có câu hỏi hoặc đề xuất, vui lòng liên hệ qua email hoặc mở Issue trên GitHub. 