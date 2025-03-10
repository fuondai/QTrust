#!/usr/bin/env python
# coding: utf-8

"""
Script chạy mô phỏng nâng cao và lưu log kết quả
"""

import os
import sys
import subprocess
import datetime
import traceback

# Tạo thư mục logs nếu chưa tồn tại
os.makedirs("run_logs", exist_ok=True)

# Tạo tên file log
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"run_logs/run_log_{timestamp}.txt"

# Command cần chạy
python_path = r"c:\users\dadad\appdata\local\programs\python\python310\python.exe"
command = [
    python_path, 
    "-m", "dqn_blockchain_sim.run_advanced_simulation",
    "--num_shards", "2",    # Giảm xuống còn 2 shard
    "--steps", "2",         # Giảm xuống còn 2 bước
    "--tx_per_step", "1",   # Giảm xuống còn 1 giao dịch/bước
    "--save_stats"
]

# Chạy command và lưu output
try:
    print(f"Đang chạy command: {' '.join(command)}")
    print(f"Log sẽ được lưu vào: {log_file}")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Command: {' '.join(command)}\n")
        f.write(f"Thời gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # Đọc từng dòng output và cả ghi vào file và hiển thị
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end='', flush=True)
            
        process.wait()
        
        f.write(f"\nExit code: {process.returncode}\n")
        
    print(f"\nĐã chạy xong với exit code: {process.returncode}")
    
except Exception as e:
    print(f"Lỗi: {e}")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\nLỗi: {e}\n")
        f.write(traceback.format_exc())
    sys.exit(1)
    
print(f"Đã lưu log tại: {log_file}") 