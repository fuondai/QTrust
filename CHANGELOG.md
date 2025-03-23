# Changelog

Tất cả những thay đổi đáng chú ý trong dự án QTrust sẽ được ghi lại trong file này.

Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
và dự án tuân thủ [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Thêm mới
- Bổ sung các biểu đồ so sánh mới trong trang README
- Cập nhật tài liệu API trong DOCUMENTATION.md
- Mở rộng hướng dẫn đóng góp trong CONTRIBUTING.md

### Sửa đổi
- Nâng cấp thuật toán phát hiện tấn công Eclipse
- Cải thiện độ chính xác của módun dự đoán tắc nghẽn

## [1.0.0] - 2025-03-23

### Thêm mới
- Phát hành phiên bản chính thức đầu tiên của QTrust
- Bộ kiểm tra tấn công với các kịch bản 51%, Sybil, Eclipse và Mixed
- Mô phỏng khả năng mở rộng với nhiều cấu hình hệ thống
- Báo cáo và biểu đồ so sánh hiệu suất tự động
- Tài liệu API đầy đủ và hướng dẫn sử dụng
- Kiểm tra đơn vị toàn diện cho tất cả các module

### Sửa đổi
- Cải thiện đáng kể hiệu suất DQN Agent
- Tối ưu hóa thuật toán định tuyến MAD-RAPID
- Giảm thiểu tài nguyên cần thiết cho mô phỏng quy mô lớn

### Sửa lỗi
- Khắc phục sự cố đồng bộ hóa dữ liệu trong học tập liên hợp
- Sửa lỗi tràn bộ nhớ trong các mô phỏng dài hạn
- Giải quyết lỗi trong phát hiện nút độc hại khi tỷ lệ nút độc hại cao

## [0.9.0] - 2025-03-15

### Thêm mới
- Thêm cơ chế HTDCM (Hierarchical Trust Data Center Mechanism)
- Tích hợp đầy đủ Federated Learning cho các Agent
- Thêm chế độ mô phỏng tấn công
- Thêm công cụ phân tích hiệu suất tự động
- Script thiết lập và cài đặt đơn giản hóa

### Sửa đổi
- Cải thiện cơ chế phần thưởng của DQN Agent
- Cập nhật thuật toán Adaptive Consensus để hỗ trợ nhiều giao thức hơn
- Tái cấu trúc pipeline dữ liệu để tăng hiệu suất

### Sửa lỗi
- Sửa lỗi trong cơ chế định tuyến trên các mạng lớn
- Khắc phục vấn đề với việc lưu trữ mô hình và tải lại

## [0.8.0] - 2025-03-01

### Thêm mới
- Thêm router MAD-RAPID cho định tuyến xuyên shard
- Thêm visualizer cho kết quả mô phỏng
- Tích hợp công cụ tracking thử nghiệm
- Thêm các chế độ mô phỏng cho các kịch bản khác nhau
- Chuẩn bị hệ thống Docker cho triển khai dễ dàng

### Sửa đổi
- Cải thiện cấu trúc dự án theo mô hình module hóa
- Tối ưu hóa hiệu suất của các mô phỏng quy mô lớn
- Cập nhật thuật toán DQN với các cải tiến hiện đại

### Sửa lỗi
- Giải quyết vấn đề với việc lưu và tải các mô hình đã huấn luyện
- Sửa lỗi trong cơ chế thưởng cho các hành động đồng thuận

## [0.7.0] - 2025-02-15

### Thêm mới
- Tạo môi trường blockchain sharding cơ bản
- Triển khai DQN Agent đầu tiên
- Cài đặt Adaptive Consensus Protocol
- Xây dựng framework mô phỏng cơ bản
- Tích hợp metrics theo dõi hiệu suất

### Sửa đổi
- Điều chỉnh cấu hình đồng thuận để cải thiện hiệu suất
- Tối ưu hóa kiến trúc DQN cho bài toán sharding

## [0.1.0] - 2025-01-15

### Thêm mới
- Khởi tạo dự án
- Thiết lập cấu trúc cơ bản của dự án
- Tạo tài liệu và README cơ bản
- Xác định kiến trúc hệ thống và module
- Thiết lập quy trình phát triển và CI/CD

[Unreleased]: https://github.com/username/qtrust/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/username/qtrust/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/username/qtrust/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/username/qtrust/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/username/qtrust/compare/v0.1.0...v0.7.0
[0.1.0]: https://github.com/username/qtrust/releases/tag/v0.1.0 