"""
Module mô phỏng giao dịch trên blockchain
"""

import time
import hashlib
import uuid
from enum import Enum
import json
from typing import Dict, Any, Optional, List


class TransactionStatus(Enum):
    """Trạng thái của giao dịch"""
    PENDING = "pending"         # Đang chờ xử lý
    CONFIRMED = "confirmed"     # Đã được xác nhận
    REJECTED = "rejected"       # Bị từ chối
    TIMEOUT = "timeout"         # Hết thời gian
    CROSS_SHARD = "cross_shard" # Đang trong quá trình xử lý xuyên mảnh


class Transaction:
    """
    Lớp mô phỏng giao dịch blockchain
    """
    
    def __init__(self, 
                 sender_id: str = "", 
                 receiver_id: str = "", 
                 amount: float = 0.0, 
                 source_shard: Optional[int] = None,
                 target_shard: Optional[int] = None,
                 value: Optional[float] = None,  # Alias for amount
                 data: Optional[Dict[str, Any]] = None, 
                 gas_price: float = 1.0,
                 gas_limit: int = 21000):
        """
        Khởi tạo giao dịch mới
        
        Args:
            sender_id: ID người gửi
            receiver_id: ID người nhận
            amount: Giá trị giao dịch
            source_shard: ID của shard nguồn
            target_shard: ID của shard đích
            value: Alias for amount (để tương thích)
            data: Dữ liệu bổ sung (nếu có)
            gas_price: Giá gas
            gas_limit: Giới hạn gas
        """
        self.transaction_id = str(uuid.uuid4())
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.amount = value if value is not None else amount  # Use value if provided
        self.data = data if data is not None else {}
        self.timestamp = time.time()
        self.gas_price = gas_price
        self.gas_limit = gas_limit
        self.status = TransactionStatus.PENDING
        self.source_shard = source_shard
        self.target_shard = target_shard
        self.confirmations = 0
        self.required_confirmations = 1  # Sẽ được cập nhật dựa trên cấu hình mạng
        self.confirmation_timestamps = []
        self.hash = self._calculate_hash()
        self.value = self.amount  # Alias for amount
    
    def _calculate_hash(self) -> str:
        """
        Tính toán hash của giao dịch
        
        Returns:
            Chuỗi hash của giao dịch
        """
        tx_dict = {
            'sender': self.sender_id,
            'receiver': self.receiver_id,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'data': self.data,
            'transaction_id': self.transaction_id
        }
        
        tx_string = json.dumps(tx_dict, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def update_status(self, new_status: TransactionStatus) -> None:
        """
        Cập nhật trạng thái giao dịch
        
        Args:
            new_status: Trạng thái mới
        """
        self.status = new_status
        
    def add_confirmation(self, validator_id: str) -> bool:
        """
        Thêm xác nhận từ validator
        
        Args:
            validator_id: ID của validator
            
        Returns:
            True nếu giao dịch đủ xác nhận, False nếu chưa đủ
        """
        if self.status != TransactionStatus.PENDING and self.status != TransactionStatus.CROSS_SHARD:
            return False
        
        self.confirmations += 1
        self.confirmation_timestamps.append(time.time())
        
        if self.confirmations >= self.required_confirmations:
            self.status = TransactionStatus.CONFIRMED
            return True
        
        return False
    
    def is_cross_shard(self) -> bool:
        """
        Kiểm tra xem giao dịch có phải là xuyên mảnh không
        
        Returns:
            True nếu là giao dịch xuyên mảnh, False nếu không phải
        """
        return self.source_shard != self.target_shard and self.source_shard is not None and self.target_shard is not None
    
    def get_latency(self) -> float:
        """
        Tính toán độ trễ của giao dịch
        
        Returns:
            Độ trễ tính bằng giây, hoặc -1 nếu giao dịch chưa hoàn thành
        """
        if self.status == TransactionStatus.CONFIRMED and self.confirmation_timestamps:
            return self.confirmation_timestamps[-1] - self.timestamp
        return -1
    
    def estimate_fee(self) -> float:
        """
        Ước tính phí giao dịch
        
        Returns:
            Phí giao dịch
        """
        data_size = len(json.dumps(self.data).encode())
        # Tính phí dựa trên kích thước dữ liệu và các tham số khác
        base_fee = self.gas_price * 21000  # Phí cơ bản
        data_fee = self.gas_price * 68 * data_size  # 68 gas cho mỗi byte dữ liệu
        return min(base_fee + data_fee, self.gas_price * self.gas_limit)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi giao dịch thành từ điển
        
        Returns:
            Từ điển biểu diễn giao dịch
        """
        return {
            'transaction_id': self.transaction_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'amount': self.amount,
            'data': self.data,
            'timestamp': self.timestamp,
            'gas_price': self.gas_price,
            'gas_limit': self.gas_limit,
            'status': self.status.value,
            'source_shard': self.source_shard,
            'target_shard': self.target_shard,
            'confirmations': self.confirmations,
            'required_confirmations': self.required_confirmations,
            'hash': self.hash,
            'latency': self.get_latency()
        }
    
    def __str__(self) -> str:
        """
        Biểu diễn chuỗi của giao dịch
        
        Returns:
            Chuỗi mô tả giao dịch
        """
        return f"Transaction(id={self.transaction_id}, sender={self.sender_id}, receiver={self.receiver_id}, amount={self.amount}, status={self.status.value})" 