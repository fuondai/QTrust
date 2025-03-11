"""
Module sinh giao dịch giả lập cho mô phỏng
"""

import random
import time
from typing import List, Dict, Any

def generate_transaction() -> Dict[str, Any]:
    """
    Sinh một giao dịch ngẫu nhiên
    
    Returns:
        Từ điển chứa thông tin giao dịch
    """
    transaction_types = ['transfer', 'smart_contract', 'token_mint', 'token_burn']
    
    transaction = {
        'id': f'tx_{int(time.time() * 1000)}_{random.randint(0, 1000000)}',
        'type': random.choice(transaction_types),
        'timestamp': time.time(),
        'amount': random.uniform(0.1, 100.0),
        'fee': random.uniform(0.01, 1.0),
        'priority': random.randint(1, 10),
        'size': random.randint(1000, 5000),  # bytes
        'complexity': random.randint(1, 5)
    }
    
    return transaction

def generate_transactions(num_transactions: int) -> List[Dict[str, Any]]:
    """
    Sinh một danh sách giao dịch ngẫu nhiên
    
    Args:
        num_transactions: Số lượng giao dịch cần sinh
        
    Returns:
        Danh sách các giao dịch
    """
    return [generate_transaction() for _ in range(num_transactions)]

def generate_batch_transactions(batch_size: int, num_batches: int) -> List[List[Dict[str, Any]]]:
    """
    Sinh nhiều batch giao dịch
    
    Args:
        batch_size: Kích thước mỗi batch
        num_batches: Số lượng batch
        
    Returns:
        Danh sách các batch giao dịch
    """
    return [generate_transactions(batch_size) for _ in range(num_batches)]

def generate_smart_contract_transaction() -> Dict[str, Any]:
    """
    Sinh giao dịch smart contract
    
    Returns:
        Từ điển chứa thông tin giao dịch smart contract
    """
    contract_types = ['token', 'defi', 'nft', 'dao']
    
    transaction = generate_transaction()
    transaction.update({
        'type': 'smart_contract',
        'contract_type': random.choice(contract_types),
        'gas_limit': random.randint(50000, 1000000),
        'gas_price': random.uniform(1, 100),
        'code_size': random.randint(5000, 50000),  # bytes
        'complexity': random.randint(3, 10)
    })
    
    return transaction

def generate_cross_shard_transaction(num_shards: int) -> Dict[str, Any]:
    """
    Sinh giao dịch xuyên shard
    
    Args:
        num_shards: Số lượng shard trong hệ thống
        
    Returns:
        Từ điển chứa thông tin giao dịch xuyên shard
    """
    transaction = generate_transaction()
    
    # Chọn ngẫu nhiên shard nguồn và đích khác nhau
    source_shard = random.randint(0, num_shards - 1)
    target_shard = random.randint(0, num_shards - 1)
    while target_shard == source_shard:
        target_shard = random.randint(0, num_shards - 1)
        
    transaction.update({
        'is_cross_shard': True,
        'source_shard': source_shard,
        'target_shard': target_shard,
        'coordination_fee': random.uniform(0.1, 2.0),
        'timeout': random.randint(30, 120)  # seconds
    })
    
    return transaction 