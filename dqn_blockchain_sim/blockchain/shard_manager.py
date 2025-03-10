"""
Module for managing blockchain shards
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .transaction import Transaction, TransactionStatus

class Shard:
    """
    Represents a blockchain shard
    """
    
    def __init__(self, shard_id: str, capacity: int = 1000):
        """
        Initialize a new shard
        
        Args:
            shard_id: Unique identifier for the shard
            capacity: Maximum number of transactions per block
        """
        self.shard_id = shard_id
        self.capacity = capacity
        self.transactions = {}  # transaction_id -> Transaction
        self.pending_transactions = []  # List of pending transaction IDs
        self.confirmed_transactions = []  # List of confirmed transaction IDs
        self.cross_shard_transactions = {}  # transaction_id -> (source_shard, target_shard)
        self.validators = []  # List of validator node IDs
        self.load = 0.0  # Current load (0-1)
        self.creation_time = time.time()
        self.last_block_time = time.time()
        self.blocks = []  # List of blocks in this shard
        
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the shard
        
        Args:
            transaction: Transaction to add
            
        Returns:
            True if added successfully, False otherwise
        """
        if transaction.transaction_id in self.transactions:
            return False
            
        self.transactions[transaction.transaction_id] = transaction
        
        if transaction.is_cross_shard():
            self.cross_shard_transactions[transaction.transaction_id] = (
                transaction.source_shard, transaction.target_shard
            )
        else:
            self.pending_transactions.append(transaction.transaction_id)
            
        # Update load
        self.update_load()
        
        return True
        
    def update_load(self) -> float:
        """
        Update and return the current load of the shard
        
        Returns:
            Current load (0-1)
        """
        # Calculate load based on pending transactions and capacity
        self.load = min(1.0, len(self.pending_transactions) / self.capacity)
        return self.load
        
    def add_validator(self, validator_id: str) -> None:
        """
        Add a validator to the shard
        
        Args:
            validator_id: ID of the validator node
        """
        if validator_id not in self.validators:
            self.validators.append(validator_id)
            
    def remove_validator(self, validator_id: str) -> bool:
        """
        Remove a validator from the shard
        
        Args:
            validator_id: ID of the validator node
            
        Returns:
            True if removed, False if not found
        """
        if validator_id in self.validators:
            self.validators.remove(validator_id)
            return True
        return False
        
    def process_transactions(self, max_transactions: int = None) -> List[str]:
        """
        Process pending transactions
        
        Args:
            max_transactions: Maximum number of transactions to process
            
        Returns:
            List of processed transaction IDs
        """
        if max_transactions is None:
            max_transactions = self.capacity
            
        to_process = self.pending_transactions[:max_transactions]
        processed = []
        
        for tx_id in to_process:
            if tx_id in self.transactions:
                tx = self.transactions[tx_id]
                
                # Simulate transaction processing
                tx.update_status(TransactionStatus.CONFIRMED)
                tx.confirmations = tx.required_confirmations
                tx.confirmation_timestamps.append(time.time())
                
                # Move from pending to confirmed
                self.pending_transactions.remove(tx_id)
                self.confirmed_transactions.append(tx_id)
                processed.append(tx_id)
                
        # Update load after processing
        self.update_load()
        
        return processed
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this shard
        
        Returns:
            Dictionary of metrics
        """
        now = time.time()
        
        # Calculate average transaction latency
        latencies = [
            self.transactions[tx_id].get_latency() 
            for tx_id in self.confirmed_transactions
            if self.transactions[tx_id].get_latency() > 0
        ]
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "shard_id": self.shard_id,
            "load": self.load,
            "pending_count": len(self.pending_transactions),
            "confirmed_count": len(self.confirmed_transactions),
            "cross_shard_count": len(self.cross_shard_transactions),
            "validator_count": len(self.validators),
            "avg_latency": avg_latency,
            "uptime": now - self.creation_time,
            "last_block_time": now - self.last_block_time
        }


class ShardManager:
    """
    Manages multiple shards in the blockchain network
    """
    
    def __init__(self, initial_shards: int = 4, shard_capacity: int = 1000):
        """
        Initialize the shard manager
        
        Args:
            initial_shards: Number of initial shards
            shard_capacity: Capacity of each shard
        """
        self.shards = {}  # shard_id -> Shard
        self.node_to_shard = {}  # node_id -> shard_id
        self.transaction_to_shard = {}  # transaction_id -> shard_id
        
        # Create initial shards
        for _ in range(initial_shards):
            shard_id = str(uuid.uuid4())
            self.shards[shard_id] = Shard(shard_id, shard_capacity)
            
    def get_shard_for_transaction(self, transaction: Transaction) -> Tuple[str, str]:
        """
        Determine which shard(s) should handle a transaction
        
        Args:
            transaction: Transaction to route
            
        Returns:
            Tuple of (source_shard_id, target_shard_id)
        """
        # Simple sharding based on sender and receiver IDs
        sender_shard = self._get_shard_for_account(transaction.sender_id)
        receiver_shard = self._get_shard_for_account(transaction.receiver_id)
        
        # Update transaction with shard information
        transaction.source_shard = sender_shard
        transaction.target_shard = receiver_shard
        
        return sender_shard, receiver_shard
        
    def _get_shard_for_account(self, account_id: str) -> str:
        """
        Determine which shard an account belongs to
        
        Args:
            account_id: Account ID
            
        Returns:
            Shard ID
        """
        # Simple hash-based sharding
        shard_ids = list(self.shards.keys())
        shard_index = hash(account_id) % len(shard_ids)
        return shard_ids[shard_index]
        
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to the appropriate shard(s)
        
        Args:
            transaction: Transaction to add
            
        Returns:
            True if added successfully
        """
        source_shard, target_shard = self.get_shard_for_transaction(transaction)
        
        # Add to source shard
        if not self.shards[source_shard].add_transaction(transaction):
            return False
            
        # If cross-shard, add to target shard as well
        if source_shard != target_shard:
            transaction.update_status(TransactionStatus.CROSS_SHARD)
            self.shards[target_shard].add_transaction(transaction)
            
        # Track which shard has this transaction
        self.transaction_to_shard[transaction.transaction_id] = source_shard
        
        return True
        
    def assign_node_to_shard(self, node_id: str, shard_id: str = None) -> str:
        """
        Assign a node to a shard
        
        Args:
            node_id: Node ID
            shard_id: Shard ID (if None, assign to least loaded shard)
            
        Returns:
            Assigned shard ID
        """
        if shard_id is None:
            # Find least loaded shard
            shard_id = min(
                self.