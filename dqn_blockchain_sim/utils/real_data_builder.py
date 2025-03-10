"""
Module cung cấp công cụ xây dựng dữ liệu thực từ blockchain Ethereum
"""

import os
import time
import json
import requests
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


class EthereumDataProcessor:
    """
    Bộ xử lý dữ liệu từ blockchain Ethereum
    """
    def __init__(self, api_key: str = None, api_url: str = None):
        """
        Khởi tạo bộ xử lý dữ liệu Ethereum
        
        Args:
            api_key: Khóa API Ethereum (Infura, Alchemy, v.v.)
            api_url: URL API tùy chỉnh (nếu không sử dụng Infura mặc định)
        """
        self.api_key = api_key
        
        # Sử dụng Infura làm mặc định nếu không cung cấp URL
        if api_url:
            self.api_url = api_url
        elif api_key:
            self.api_url = f"https://mainnet.infura.io/v3/{api_key}"
        else:
            self.api_url = None
            
        # Lưu trữ cache để tránh requests lặp lại
        self.cache = {}
        self.cache_expiry = 3600  # 1 giờ
        
    def _make_request(self, method: str, params: List) -> Dict:
        """
        Thực hiện một yêu cầu JSON-RPC đến API Ethereum
        
        Args:
            method: Phương thức JSON-RPC
            params: Danh sách tham số
            
        Returns:
            Dữ liệu phản hồi
        """
        if not self.api_url:
            raise ValueError("Không có URL API hoặc API key")
            
        # Tạo cache key
        cache_key = f"{method}:{json.dumps(params)}"
        
        # Kiểm tra cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_expiry:
                return cached_data
                
        # Chuẩn bị payload JSON-RPC
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": int(time.time() * 1000)
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Thực hiện request với xử lý lỗi và thử lại
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    raise ValueError(f"Lỗi JSON-RPC: {data['error']}")
                    
                # Lưu vào cache
                self.cache[cache_key] = (data, time.time())
                return data
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1 * (attempt + 1))  # Thử lại với backoff
                
        return None  # Không bao giờ đến đây
        
    def get_latest_block_number(self) -> int:
        """
        Lấy số khối mới nhất trên blockchain
        
        Returns:
            Số khối mới nhất
        """
        data = self._make_request("eth_blockNumber", [])
        return int(data["result"], 16)
        
    def get_block_by_number(self, block_number: int, include_transactions: bool = True) -> Dict:
        """
        Lấy thông tin khối theo số khối
        
        Args:
            block_number: Số khối
            include_transactions: Lấy đầy đủ thông tin giao dịch hay chỉ hash
            
        Returns:
            Dữ liệu khối
        """
        # Chuyển đổi số khối sang hex
        block_hex = hex(block_number)
        
        data = self._make_request("eth_getBlockByNumber", [block_hex, include_transactions])
        return data["result"]
        
    def get_transaction_by_hash(self, tx_hash: str) -> Dict:
        """
        Lấy thông tin giao dịch theo hash
        
        Args:
            tx_hash: Hash của giao dịch
            
        Returns:
            Dữ liệu giao dịch
        """
        data = self._make_request("eth_getTransactionByHash", [tx_hash])
        return data["result"]
        
    def extract_transactions(self, start_block: int, end_block: int) -> List[Dict]:
        """
        Trích xuất danh sách giao dịch từ một dải khối
        
        Args:
            start_block: Khối bắt đầu
            end_block: Khối kết thúc
            
        Returns:
            Danh sách giao dịch
        """
        transactions = []
        
        # Giới hạn truy vấn để tránh quá tải
        max_blocks = min(end_block - start_block + 1, 100)
        actual_end_block = start_block + max_blocks - 1
        
        print(f"Đang trích xuất giao dịch từ khối {start_block} đến {actual_end_block}...")
        
        for block_number in range(start_block, actual_end_block + 1):
            try:
                # Lấy thông tin khối
                block_data = self.get_block_by_number(block_number, True)
                
                if not block_data or "transactions" not in block_data:
                    continue
                    
                # Thêm thông tin timestamp từ khối vào mỗi giao dịch
                block_timestamp = int(block_data["timestamp"], 16)
                
                # Xử lý từng giao dịch trong khối
                for tx in block_data["transactions"]:
                    # Chuyển đổi các giá trị hex sang decimal
                    processed_tx = {
                        "hash": tx["hash"],
                        "blockNumber": int(tx["blockNumber"], 16),
                        "from": tx["from"],
                        "to": tx["to"] if tx["to"] else "0x",  # Đối với contract creation, to = null
                        "value": float(int(tx["value"], 16)) / 1e18,  # Chuyển đổi từ wei sang ether
                        "gas": int(tx["gas"], 16),
                        "gasPrice": int(tx["gasPrice"], 16) if "gasPrice" in tx else 0,
                        "nonce": int(tx["nonce"], 16),
                        "timeStamp": block_timestamp,
                        "input": tx["input"]
                    }
                    
                    transactions.append(processed_tx)
                    
                # Giới hạn số lượng giao dịch để tránh quá tải
                if len(transactions) >= 1000:
                    break
                    
            except Exception as e:
                print(f"Lỗi khi xử lý khối {block_number}: {e}")
                continue
                
        return transactions
        
    def save_transactions(self, transactions: List[Dict], filepath: str):
        """
        Lưu danh sách giao dịch vào file
        
        Args:
            transactions: Danh sách giao dịch
            filepath: Đường dẫn file
        """
        # Xác định định dạng file dựa trên phần mở rộng
        _, ext = os.path.splitext(filepath)
        
        # Tạo DataFrame từ danh sách giao dịch
        df = pd.DataFrame(transactions)
        
        # Lưu theo định dạng tương ứng
        if ext.lower() == '.csv':
            df.to_csv(filepath, index=False)
        elif ext.lower() == '.json':
            df.to_json(filepath, orient='records')
        elif ext.lower() == '.parquet':
            df.to_parquet(filepath, index=False)
        else:
            # Mặc định lưu dưới dạng CSV
            df.to_csv(f"{filepath}.csv", index=False)
            
        print(f"Đã lưu {len(transactions)} giao dịch vào {filepath}")
        
    def analyze_transaction_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Phân tích mẫu giao dịch từ dữ liệu
        
        Args:
            transactions: Danh sách giao dịch
            
        Returns:
            Thống kê và mẫu giao dịch
        """
        if not transactions:
            return {"error": "Không có dữ liệu giao dịch"}
            
        # Chuyển đổi sang DataFrame
        df = pd.DataFrame(transactions)
        
        # Thống kê cơ bản
        stats = {
            "total_transactions": len(df),
            "total_value": df["value"].sum(),
            "avg_value": df["value"].mean(),
            "avg_gas": df["gas"].mean() if "gas" in df.columns else 0,
            "unique_senders": df["from"].nunique(),
            "unique_receivers": df["to"].nunique()
        }
        
        # Phân tích phân phối giá trị
        value_bins = [0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, float('inf')]
        value_labels = ['dust', 'very_small', 'small', 'medium', 'large', 'very_large', 'huge']
        
        df['value_group'] = pd.cut(df['value'], bins=value_bins, labels=value_labels)
        value_distribution = df['value_group'].value_counts().to_dict()
        stats["value_distribution"] = value_distribution
        
        # Phân tích hợp đồng phổ biến
        if 'to' in df.columns:
            popular_contracts = df["to"].value_counts().head(10).to_dict()
            stats["popular_contracts"] = popular_contracts
            
        # Phân tích dữ liệu input
        if 'input' in df.columns:
            # Phân loại giao dịch thông thường và giao dịch hợp đồng
            has_input = df['input'].str.len() > 3  # "0x" + ít nhất 1 byte
            contract_txs = df[has_input]
            stats["contract_interaction_ratio"] = len(contract_txs) / len(df)
            
        # Phân tích phân phối theo thời gian (nếu có dữ liệu timestamp)
        if 'timeStamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timeStamp'], unit='s').dt.hour
            time_distribution = df['hour'].value_counts().sort_index().to_dict()
            stats["time_distribution"] = time_distribution
            
        return stats
        
    def extract_shard_simulation_data(self, transactions: List[Dict], num_shards: int = 8) -> Dict[str, Any]:
        """
        Chuyển đổi dữ liệu giao dịch thành dữ liệu mô phỏng shard
        
        Args:
            transactions: Danh sách giao dịch
            num_shards: Số lượng shard trong mô phỏng
            
        Returns:
            Dữ liệu mô phỏng với phân bổ giao dịch vào các shard
        """
        if not transactions:
            return {"error": "Không có dữ liệu giao dịch"}
            
        # Chuyển đổi sang DataFrame
        df = pd.DataFrame(transactions)
        
        # Mã hoá địa chỉ thành shard
        def address_to_shard(address):
            if not address or address == "0x":
                return 0
            # Sử dụng 2 byte cuối của địa chỉ để xác định shard
            return int(address[-4:], 16) % num_shards
            
        df['source_shard'] = df['from'].apply(address_to_shard)
        df['target_shard'] = df['to'].apply(address_to_shard)
        
        # Tạo dữ liệu giao dịch cho mô phỏng
        tx_data = []
        for _, row in df.iterrows():
            tx = {
                'source_shard': row['source_shard'],
                'target_shard': row['target_shard'],
                'value': row['value'],
                'gas_limit': row['gas'] if 'gas' in row else 21000,
                'timestamp': row['timeStamp'] if 'timeStamp' in row else 0,
                'is_contract': len(row.get('input', '0x')) > 3 if 'input' in row else False
            }
            tx_data.append(tx)
            
        # Thống kê giao dịch theo shard
        shard_stats = {}
        for i in range(num_shards):
            # Giao dịch bắt nguồn từ shard này
            outgoing = df[df['source_shard'] == i]
            # Giao dịch đến shard này
            incoming = df[df['target_shard'] == i]
            # Giao dịch trong cùng shard
            internal = df[(df['source_shard'] == i) & (df['target_shard'] == i)]
            # Giao dịch xuyên shard
            cross_shard_out = df[(df['source_shard'] == i) & (df['target_shard'] != i)]
            
            shard_stats[i] = {
                'total_outgoing': len(outgoing),
                'total_incoming': len(incoming),
                'internal_tx': len(internal),
                'cross_shard_tx': len(cross_shard_out),
                'cross_shard_ratio': len(cross_shard_out) / max(1, len(outgoing))
            }
            
        # Tạo ma trận kết nối giữa các shard
        connection_matrix = {}
        for i in range(num_shards):
            connection_matrix[i] = {}
            for j in range(num_shards):
                if i != j:
                    # Số giao dịch từ shard i đến shard j
                    count = len(df[(df['source_shard'] == i) & (df['target_shard'] == j)])
                    connection_matrix[i][j] = count
                    
        return {
            'transactions': tx_data,
            'shard_stats': shard_stats,
            'connection_matrix': connection_matrix
        } 

class RealDataBuilder:
    """
    Xây dựng dữ liệu mô phỏng từ dữ liệu blockchain thực
    """
    def __init__(self, api_key: str = None, data_dir: str = "data"):
        """
        Khởi tạo builder
        
        Args:
            api_key: Khóa API cho Ethereum
            data_dir: Thư mục lưu dữ liệu
        """
        self.api_key = api_key
        self.data_dir = data_dir
        self.processor = EthereumDataProcessor(api_key=api_key)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(data_dir, exist_ok=True)
        
    def fetch_real_data(self, start_block: int = None, num_blocks: int = 10) -> List[Dict]:
        """
        Lấy dữ liệu từ blockchain Ethereum
        
        Args:
            start_block: Block bắt đầu (None = block mới nhất - num_blocks)
            num_blocks: Số block cần lấy
            
        Returns:
            Danh sách các giao dịch đã lấy
        """
        # Nếu không chỉ định block bắt đầu, dùng block mới nhất
        if start_block is None:
            latest_block = self.processor.get_latest_block_number()
            start_block = max(1, latest_block - num_blocks)
        
        end_block = start_block + num_blocks
        
        print(f"Lấy dữ liệu từ block {start_block} đến {end_block}...")
        transactions = self.processor.extract_transactions(start_block, end_block)
        
        # Lưu dữ liệu
        timestamp = int(time.time())
        filepath = os.path.join(self.data_dir, f"ethereum_data_{timestamp}.json")
        self.processor.save_transactions(transactions, filepath)
        
        return transactions
        
    def build_transactions(self, num_shards: int = 8, use_cached: bool = True) -> List[Dict]:
        """
        Xây dựng tập dữ liệu giao dịch cho mô phỏng
        
        Args:
            num_shards: Số lượng shard cần mô phỏng
            use_cached: Sử dụng dữ liệu đã lưu nếu có
            
        Returns:
            Danh sách giao dịch đã xử lý
        """
        # Kiểm tra dữ liệu đã lưu
        if use_cached:
            cached_files = [f for f in os.listdir(self.data_dir) 
                          if f.startswith("ethereum_data_") and f.endswith(".json")]
            
            if cached_files:
                # Dùng file mới nhất
                latest_file = sorted(cached_files)[-1]
                filepath = os.path.join(self.data_dir, latest_file)
                
                print(f"Sử dụng dữ liệu đã lưu từ {filepath}")
                with open(filepath, 'r') as f:
                    transactions = json.load(f)
                    
                # Nếu đã có đủ dữ liệu (ít nhất 100 giao dịch), sử dụng luôn
                if len(transactions) >= 100:
                    return self._process_transactions(transactions, num_shards)
        
        # Nếu không có dữ liệu đã lưu hoặc không đủ, lấy mới
        transactions = self.fetch_real_data(num_blocks=20)
        return self._process_transactions(transactions, num_shards)
    
    def _process_transactions(self, transactions: List[Dict], num_shards: int) -> List[Dict]:
        """
        Xử lý các giao dịch Ethereum thành định dạng phù hợp cho mô phỏng
        
        Args:
            transactions: Danh sách giao dịch thô từ Ethereum
            num_shards: Số lượng shard cần mô phỏng
            
        Returns:
            Danh sách giao dịch đã xử lý
        """
        processed_tx = []
        simulation_data = self.processor.extract_shard_simulation_data(transactions, num_shards)
        
        for i, tx in enumerate(simulation_data['transactions']):
            processed_tx.append({
                'transaction_id': f"tx_{i}",
                'from_shard': tx['from_shard'],
                'to_shard': tx['to_shard'],
                'value': tx['value'] / 1e18,  # Convert từ wei sang eth
                'gas': tx['gas'],
                'timestamp': tx['timestamp'],
                'type': 'cross_shard' if tx['from_shard'] != tx['to_shard'] else 'intra_shard',
                'data_size': len(tx.get('input', '0x')) // 2,  # Kích thước dữ liệu (bytes)
                'original_tx': tx['hash']  # Lưu hash gốc để tham chiếu
            })
            
        return processed_tx 