#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script cài đặt tự động cho dự án QTrust.
Tạo môi trường, cài đặt thư viện và chuẩn bị thư mục.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_header(text):
    """In header đẹp mắt."""
    print("\n" + "="*80)
    print(f"== {text}")
    print("=" * 80 + "\n")

def run_command(command, description=None):
    """Chạy lệnh shell và hiển thị kết quả."""
    if description:
        print(f"\n>> {description}...")
    
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Lỗi: {result.stderr}")
        return False
    
    print(result.stdout)
    return True

def create_virtual_env():
    """Tạo môi trường ảo."""
    print_header("Tạo môi trường ảo Python")
    
    # Kiểm tra Python version
    python_version = platform.python_version()
    print(f"Python version: {python_version}")
    
    # Xác định lệnh tạo venv phù hợp
    if sys.platform.startswith('win'):
        venv_command = "python -m venv venv"
        activate_command = ".\\venv\\Scripts\\activate"
    else:
        venv_command = "python3 -m venv venv"
        activate_command = "source venv/bin/activate"
    
    # Tạo venv
    if os.path.exists("venv"):
        choice = input("Môi trường ảo đã tồn tại. Bạn có muốn tạo lại không? (y/n): ")
        if choice.lower() == 'y':
            shutil.rmtree("venv")
        else:
            print("Bỏ qua tạo môi trường ảo.")
            return activate_command
    
    run_command(venv_command, "Tạo môi trường ảo")
    print(f"Môi trường ảo đã được tạo. Kích hoạt bằng lệnh: {activate_command}")
    
    return activate_command

def install_dependencies(activate_command):
    """Cài đặt dependencies."""
    print_header("Cài đặt thư viện phụ thuộc")
    
    if sys.platform.startswith('win'):
        command = f"{activate_command} && pip install -r requirements.txt"
    else:
        command = f"{activate_command} && pip install -r requirements.txt"
    
    run_command(command, "Cài đặt dependencies từ requirements.txt")
    
    # Cài đặt gói trong chế độ phát triển
    if sys.platform.startswith('win'):
        command = f"{activate_command} && pip install -e ."
    else:
        command = f"{activate_command} && pip install -e ."
    
    run_command(command, "Cài đặt gói QTrust trong chế độ phát triển")

def create_directories():
    """Tạo các thư mục cần thiết."""
    print_header("Tạo các thư mục cần thiết")
    
    directories = [
        "models",
        "results",
        "results_extended",
        "results_large",
        "results_attack",
        "results_comparison",
        "charts"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Đã tạo thư mục: {directory}")

def run_tests(activate_command):
    """Chạy tests."""
    print_header("Chạy tests")
    
    if sys.platform.startswith('win'):
        command = f"{activate_command} && python -m pytest qtrust/tests/"
    else:
        command = f"{activate_command} && python -m pytest qtrust/tests/"
    
    run_command(command, "Chạy unit tests")

def print_next_steps():
    """In hướng dẫn các bước tiếp theo."""
    print_header("THIẾT LẬP HOÀN TẤT")
    
    print("""
QTrust đã được cài đặt thành công! Các bước tiếp theo:

1. Kích hoạt môi trường ảo:
   - Windows: .\\venv\\Scripts\\activate
   - Linux/Mac: source venv/bin/activate

2. Chạy mô phỏng cơ bản:
   - python main.py

3. Xem tài liệu:
   - QTrust.pdf

4. Chạy mô phỏng với tham số tùy chỉnh:
   - python main.py --num-shards 8 --episodes 100 --eval-interval 10

Để biết thêm thông tin, hãy tham khảo README.md
""")

def main():
    """Hàm chính."""
    print_header("THIẾT LẬP MÔI TRƯỜNG CHO QTRUST")
    
    # Kiểm tra xem đang ở thư mục gốc của dự án chưa
    if not os.path.exists("setup.py") or not os.path.exists("requirements.txt"):
        print("Lỗi: Script này phải được chạy từ thư mục gốc của dự án QTrust.")
        sys.exit(1)
    
    # Tạo môi trường ảo
    activate_command = create_virtual_env()
    
    # Cài đặt dependencies
    install_dependencies(activate_command)
    
    # Tạo thư mục
    create_directories()
    
    # Chạy tests
    run_tests(activate_command)
    
    # In hướng dẫn
    print_next_steps()

if __name__ == "__main__":
    main() 