"""
Script tạo giao diện người dùng đơn giản để chạy các phân tích
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime
import subprocess

# Thêm thư mục gốc vào đường dẫn
PYTHON_PATH = r"C:\Users\dadad\AppData\Local\Programs\Python\Python310\python.exe"
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Cấu hình matplotlib để không sử dụng interactive mode
import matplotlib
matplotlib.use('Agg')

# Nhập các module phân tích
try:
    from experiments.benchmark_runner import run_comparative_benchmarks, analyze_results
    from experiments.consensus_comparison import ConsensusSimulator
    from experiments.performance_analysis import PerformanceAnalyzer
    from experiments.generate_report import ReportGenerator
    
    IMPORT_ERROR = None
except ImportError as e:
    IMPORT_ERROR = str(e)


class RedirectText:
    """Lớp chuyển hướng đầu ra sang widget Text"""
    
    def __init__(self, text_widget):
        """Khởi tạo đối tượng chuyển hướng đầu ra
        
        Args:
            text_widget: Widget Text để hiển thị đầu ra
        """
        self.text_widget = text_widget
        self.buffer = []
        
    def write(self, string):
        """Ghi chuỗi vào widget
        
        Args:
            string: Chuỗi cần ghi
        """
        self.buffer.append(string)
        # Chỉ cập nhật giao diện sau mỗi dòng hoàn chỉnh
        if string.endswith('\n'):
            self.flush()
        
    def flush(self):
        """Đẩy buffer ra widget"""
        if self.buffer:
            msg = ''.join(self.buffer)
            self.text_widget.insert(tk.END, msg)
            self.text_widget.see(tk.END)
            self.text_widget.update()
            self.buffer = []


class AnalysisGUI:
    """Giao diện người dùng đồ họa để chạy các phân tích"""
    
    def __init__(self, root):
        """
        Khởi tạo giao diện
        
        Args:
            root: Cửa sổ gốc Tkinter
        """
        self.root = root
        self.root.title("DQN Blockchain Analysis Tool")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)
        
        self.create_widgets()
        self.setup_layout()
        
        # Kiểm tra lỗi nhập module
        if IMPORT_ERROR:
            messagebox.showerror("Import Error", f"Không thể nhập các module phân tích:\n{IMPORT_ERROR}")
            self.log(f"LỖI: {IMPORT_ERROR}")
            self.log("Hãy đảm bảo rằng tất cả các thư viện phụ thuộc đã được cài đặt.")
        
        # Biến theo dõi luồng phân tích
        self.analysis_thread = None
        self.is_running = False
    
    def create_widgets(self):
        """Tạo các widget cho giao diện"""
        # Frame chính
        self.main_frame = ttk.Frame(self.root, padding=10)
        
        # Notebook (tab control)
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Tab cấu hình
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Cấu Hình")
        
        # Tab log
        self.log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.log_tab, text="Log")
        
        # Tab giới thiệu
        self.about_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.about_tab, text="Giới Thiệu")
        
        # Widgets trong tab cấu hình
        self.create_config_widgets()
        
        # Widgets trong tab log
        self.create_log_widgets()
        
        # Widgets trong tab giới thiệu
        self.create_about_widgets()
        
        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        
        # Thanh tiến trình
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        
        # Các nút điều khiển
        self.control_frame = ttk.Frame(self.main_frame)
        
        self.run_button = ttk.Button(self.control_frame, text="Chạy Phân Tích", command=self.run_analysis)
        self.stop_button = ttk.Button(self.control_frame, text="Dừng", command=self.stop_analysis, state=tk.DISABLED)
        self.open_report_button = ttk.Button(self.control_frame, text="Mở Báo Cáo", command=self.open_report)
        self.exit_button = ttk.Button(self.control_frame, text="Thoát", command=self.root.destroy)
    
    def create_config_widgets(self):
        """Tạo các widget cho tab cấu hình"""
        # Frame chứa cấu hình
        config_frame = ttk.LabelFrame(self.config_tab, text="Cấu Hình Phân Tích", padding=10)
        config_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Các tùy chọn phân tích
        self.run_benchmark_var = tk.BooleanVar(value=True)
        self.run_consensus_var = tk.BooleanVar(value=True)
        self.run_performance_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(config_frame, text="Chạy Benchmark", variable=self.run_benchmark_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(config_frame, text="Chạy So Sánh Đồng Thuận", variable=self.run_consensus_var).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(config_frame, text="Chạy Phân Tích Hiệu Suất", variable=self.run_performance_var).pack(anchor=tk.W, pady=2)
        
        # Tham số cấu hình
        param_frame = ttk.LabelFrame(config_frame, text="Tham Số", padding=10)
        param_frame.pack(fill=tk.X, pady=10)
        
        # Số lượng cấu hình benchmark
        ttk.Label(param_frame, text="Số lượng cấu hình benchmark:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.num_configs_var = tk.IntVar(value=3)
        ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.num_configs_var, width=5).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        # Thư mục đầu ra
        ttk.Label(param_frame, text="Thư mục đầu ra:").grid(row=1, column=0, sticky=tk.W, pady=2)
        
        output_frame = ttk.Frame(param_frame)
        output_frame.grid(row=1, column=1, sticky=tk.W, pady=2)
        
        self.output_dir_var = tk.StringVar(value="analysis_results")
        ttk.Entry(output_frame, textvariable=self.output_dir_var, width=20).pack(side=tk.LEFT)
        ttk.Button(output_frame, text="...", width=3, command=self.select_output_dir).pack(side=tk.LEFT, padx=2)
    
    def create_log_widgets(self):
        """Tạo các widget cho tab log"""
        # Text widget cuộn để hiển thị log
        log_frame = ttk.Frame(self.log_tab, padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Chuyển hướng đầu ra
        self.stdout_redirector = RedirectText(self.log_text)
        self.stderr_redirector = RedirectText(self.log_text)
    
    def create_about_widgets(self):
        """Tạo các widget cho tab giới thiệu"""
        about_frame = ttk.Frame(self.about_tab, padding=20)
        about_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tiêu đề
        title_label = ttk.Label(about_frame, text="DQN Blockchain Analysis Tool", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Phiên bản
        version_label = ttk.Label(about_frame, text="Version 1.0")
        version_label.pack()
        
        # Mô tả
        desc_text = """
        Công cụ này cung cấp giao diện đồ họa để chạy các phân tích trên hệ thống blockchain tối ưu hóa DQN.
        
        Các phân tích bao gồm:
        - Benchmark: So sánh hiệu suất với các cấu hình khác nhau
        - So sánh đồng thuận: So sánh các phương pháp đồng thuận khác nhau
        - Phân tích hiệu suất: Phân tích chi tiết hiệu suất của hệ thống
        
        Kết quả sẽ được tổng hợp trong một báo cáo PDF.
        """
        
        desc_label = ttk.Label(about_frame, text=desc_text, wraplength=500, justify=tk.CENTER)
        desc_label.pack(pady=10)
        
        # GitHub link
        github_label = ttk.Label(about_frame, text="GitHub: https://github.com/username/dqn_blockchain_sim", foreground="blue", cursor="hand2")
        github_label.pack()
        github_label.bind("<Button-1>", lambda e: self.open_url("https://github.com/username/dqn_blockchain_sim"))
    
    def setup_layout(self):
        """Thiết lập bố cục tổng thể"""
        # Đặt các widget vào bố cục
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Đặt các nút điều khiển
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.open_report_button.pack(side=tk.LEFT, padx=5)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # Đặt thanh tiến trình
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Đặt thanh trạng thái
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
    
    def log(self, message):
        """
        Ghi thông điệp vào log
        
        Args:
            message: Thông điệp cần ghi
        """
        timestamp = datetime.now().strftime("[%H:%M:%S] ")
        self.stdout_redirector.write(timestamp + message + "\n")
    
    def select_output_dir(self):
        """Mở hộp thoại chọn thư mục đầu ra"""
        dir_path = filedialog.askdirectory(title="Chọn thư mục đầu ra")
        if dir_path:
            self.output_dir_var.set(dir_path)
    
    def update_status(self, status, progress=None):
        """
        Cập nhật trạng thái và tiến trình
        
        Args:
            status: Thông điệp trạng thái
            progress: Giá trị tiến trình (0-100)
        """
        self.status_var.set(status)
        if progress is not None:
            self.progress_var.set(progress)
    
    def toggle_controls(self, running=False):
        """
        Bật/tắt các điều khiển dựa trên trạng thái chạy
        
        Args:
            running: True nếu đang chạy phân tích
        """
        self.is_running = running
        
        if running:
            self.run_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.NORMAL)
            self.notebook.tab(0, state=tk.DISABLED)  # Disable config tab
        else:
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.notebook.tab(0, state=tk.NORMAL)  # Enable config tab
    
    def run_analysis(self):
        """Chạy phân tích trong một luồng riêng biệt"""
        if self.is_running:
            return
            
        # Tạo thư mục đầu ra
        output_dir = self.output_dir_var.get()
        if not output_dir:
            messagebox.showerror("Lỗi", "Vui lòng chọn thư mục đầu ra")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Chuyển đến tab log
        self.notebook.select(1)
        
        # Chuyển hướng đầu ra
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        # Bật trạng thái chạy
        self.toggle_controls(running=True)
        
        # Cấu hình phân tích
        config = {
            "run_benchmark": self.run_benchmark_var.get(),
            "run_consensus": self.run_consensus_var.get(),
            "run_performance": self.run_performance_var.get(),
            "num_configs": self.num_configs_var.get(),
            "output_dir": output_dir
        }
        
        # Tạo và bắt đầu luồng phân tích
        self.analysis_thread = threading.Thread(target=self.analysis_worker, args=(config,))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
    
    def analysis_worker(self, config):
        """
        Hàm thực hiện phân tích trong luồng riêng biệt
        
        Args:
            config: Cấu hình phân tích
        """
        try:
            # Bắt đầu đo thời gian
            start_time = time.time()
            
            # In thông tin bắt đầu
            self.log(f"Bắt đầu phân tích với cấu hình: {config}")
            self.log(f"Thời gian bắt đầu: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            
            # Tạo các thư mục con
            benchmark_dir = os.path.join(config["output_dir"], "benchmark_results")
            consensus_dir = os.path.join(config["output_dir"], "consensus_comparison")
            performance_dir = os.path.join(config["output_dir"], "performance_analysis")
            report_dir = os.path.join(config["output_dir"], "final_report")
            
            os.makedirs(benchmark_dir, exist_ok=True)
            os.makedirs(consensus_dir, exist_ok=True)
            os.makedirs(performance_dir, exist_ok=True)
            os.makedirs(report_dir, exist_ok=True)
            
            total_steps = sum([config["run_benchmark"], config["run_consensus"], config["run_performance"], True])  # +1 for report
            current_step = 0
            
            # Chạy benchmark
            if config["run_benchmark"] and not self.is_running:
                return
                
            if config["run_benchmark"]:
                self.update_status("Đang chạy benchmark...", progress=(current_step / total_steps) * 100)
                self.log("\n" + "="*50)
                self.log("Chạy benchmarks...")
                self.log("="*50)
                
                # Chạy benchmark
                benchmarks = run_comparative_benchmarks()
                analyze_results(benchmarks, output_dir=benchmark_dir)
                
                self.log("Benchmarks hoàn thành.")
                current_step += 1
            
            # Chạy so sánh đồng thuận
            if config["run_consensus"] and not self.is_running:
                return
                
            if config["run_consensus"]:
                self.update_status("Đang chạy so sánh đồng thuận...", progress=(current_step / total_steps) * 100)
                self.log("\n" + "="*50)
                self.log("Chạy so sánh các phương pháp đồng thuận...")
                self.log("="*50)
                
                # Khởi tạo simulator
                simulator = ConsensusSimulator(output_dir=consensus_dir)
                
                # Tham số cho phương pháp đồng thuận
                consensus_config = {
                    'num_shards': 4,
                    'num_nodes': 20,
                    'num_validators': 10,
                    'num_transactions': 100,
                    'block_size': 10
                }
                
                # Chạy so sánh
                protocols = ["PoW", "PoS", "PBFT", "ACSC"]
                simulator.run_comparison(protocols=protocols, config=consensus_config)
                
                # Tạo biểu đồ
                simulator.visualize_comparison()
                
                self.log("So sánh các phương pháp đồng thuận hoàn thành.")
                current_step += 1
            
            # Chạy phân tích hiệu suất
            if config["run_performance"] and not self.is_running:
                return
                
            if config["run_performance"]:
                self.update_status("Đang chạy phân tích hiệu suất...", progress=(current_step / total_steps) * 100)
                self.log("\n" + "="*50)
                self.log("Chạy phân tích hiệu suất...")
                self.log("="*50)
                
                # Khởi tạo analyzer
                analyzer = PerformanceAnalyzer(output_dir=performance_dir)
                
                # Tạo các cấu hình mô phỏng
                performance_configs = []
                
                # Cấu hình với và không có DQN (2 cấu hình đơn giản)
                for use_dqn in [True, False]:
                    performance_config = {
                        'name': f"dqn_analysis_{'with' if use_dqn else 'without'}",
                        'num_shards': 4,
                        'num_steps': 20,
                        'tx_per_step': 10,
                        'use_dqn': use_dqn
                    }
                    performance_configs.append(performance_config)
                
                # Chạy phân tích
                analyzer.run_analysis(performance_configs)
                
                self.log("Phân tích hiệu suất hoàn thành.")
                current_step += 1
            
            # Tạo báo cáo tổng hợp
            if not self.is_running:
                return
                
            self.update_status("Đang tạo báo cáo tổng hợp...", progress=(current_step / total_steps) * 100)
            self.log("\n" + "="*50)
            self.log("Tạo báo cáo tổng hợp...")
            self.log("="*50)
            
            # Khởi tạo generator
            generator = ReportGenerator(output_dir=report_dir)
            
            # Đặt các thư mục dữ liệu
            generator.benchmark_dir = benchmark_dir
            generator.consensus_dir = consensus_dir
            generator.performance_dir = performance_dir
            
            # Tạo báo cáo
            report_file = generator.generate_report()
            
            self.log(f"Báo cáo đã được tạo: {report_file}")
            
            # Tính thời gian chạy
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            self.log("\n" + "="*50)
            self.log(f"Tất cả các phân tích đã hoàn thành trong: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            self.log("="*50)
            
            # Cập nhật trạng thái
            self.update_status("Phân tích hoàn thành", progress=100)
            
            # Hiển thị thông báo
            self.root.after(0, lambda: messagebox.showinfo("Hoàn thành", "Phân tích đã hoàn thành!"))
        
        except Exception as e:
            # Ghi log lỗi
            self.log(f"ERROR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            # Hiển thị thông báo lỗi
            self.root.after(0, lambda: messagebox.showerror("Lỗi", f"Đã xảy ra lỗi trong quá trình phân tích:\n{str(e)}"))
            
            # Cập nhật trạng thái
            self.update_status("Đã xảy ra lỗi")
        
        finally:
            # Khôi phục đầu ra
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            
            # Tắt trạng thái chạy
            self.root.after(0, lambda: self.toggle_controls(running=False))
    
    def stop_analysis(self):
        """Dừng phân tích"""
        if not self.is_running:
            return
            
        # Đặt cờ dừng
        self.is_running = False
        
        # Ghi log
        self.log("Đang dừng phân tích...")
        
        # Cập nhật trạng thái
        self.update_status("Đang dừng phân tích...")
    
    def open_report(self):
        """Mở báo cáo PDF gần nhất"""
        output_dir = self.output_dir_var.get()
        report_dir = os.path.join(output_dir, "final_report")
        report_file = os.path.join(report_dir, "final_report.pdf")
        
        if not os.path.exists(report_file):
            messagebox.showerror("Lỗi", "Báo cáo không tồn tại")
            return
            
        # Mở file PDF bằng ứng dụng mặc định
        try:
            if sys.platform == 'win32':
                os.startfile(report_file)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(['open', report_file])
            else:  # Linux
                subprocess.call(['xdg-open', report_file])
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở báo cáo: {str(e)}")
    
    def open_url(self, url):
        """
        Mở URL trong trình duyệt mặc định
        
        Args:
            url: URL cần mở
        """
        import webbrowser
        webbrowser.open(url)


def main():
    """Hàm chính để khởi chạy ứng dụng"""
    # Khởi tạo Tkinter
    root = tk.Tk()
    
    # Tạo giao diện
    app = AnalysisGUI(root)
    
    # Chạy ứng dụng
    root.mainloop()


if __name__ == "__main__":
    main() 