"""
Module định nghĩa các giao thức đồng thuận cơ bản
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import random
import math

class ConsensusProtocol:
    """
    Lớp cơ sở cho các giao thức đồng thuận
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Khởi tạo giao thức đồng thuận
        
        Args:
            name: Tên giao thức
            config: Cấu hình cho giao thức
        """
        self.name = name
        self.config = config if config else {}
        
        # Thống kê hiệu suất
        self.stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'avg_latency': 0,
            'energy_usage': 0
        }
        
    def validate_transaction(self, transaction: Any) -> bool:
        """
        Xác thực một giao dịch
        
        Args:
            transaction: Giao dịch cần xác thực
            
        Returns:
            True nếu giao dịch hợp lệ, False nếu không
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")
        
    def process_transaction(self, transaction: Any) -> Tuple[bool, float, float]:
        """
        Xử lý một giao dịch
        
        Args:
            transaction: Giao dịch cần xử lý
            
        Returns:
            (success, latency, energy): Kết quả xử lý, độ trễ và năng lượng tiêu thụ
        """
        raise NotImplementedError("Các lớp con phải triển khai phương thức này")
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy thống kê hiệu suất của giao thức
        
        Returns:
            Từ điển chứa các thống kê
        """
        return self.stats
        
    def reset_statistics(self):
        """
        Đặt lại thống kê về giá trị mặc định
        """
        self.stats = {
            'total_transactions': 0,
            'successful_transactions': 0,
            'failed_transactions': 0,
            'avg_latency': 0,
            'energy_usage': 0
        }
        
    def update_statistics(self, success: bool, latency: float, energy: float):
        """
        Cập nhật thống kê sau khi xử lý giao dịch
        
        Args:
            success: Giao dịch thành công hay không
            latency: Độ trễ xử lý
            energy: Năng lượng tiêu thụ
        """
        self.stats['total_transactions'] += 1
        
        if success:
            self.stats['successful_transactions'] += 1
        else:
            self.stats['failed_transactions'] += 1
            
        # Cập nhật độ trễ trung bình
        n = self.stats['total_transactions']
        old_avg = self.stats['avg_latency']
        self.stats['avg_latency'] = (old_avg * (n-1) + latency) / n
        
        # Cộng dồn năng lượng tiêu thụ
        self.stats['energy_usage'] += energy 