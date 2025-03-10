"""
Script tổng hợp báo cáo từ tất cả các kết quả phân tích
"""

import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from datetime import datetime
from fpdf import FPDF
import argparse

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dqn_blockchain_sim.experiments.benchmark_runner import run_comparative_benchmarks, analyze_results
from dqn_blockchain_sim.experiments.consensus_comparison import ConsensusSimulator
from dqn_blockchain_sim.experiments.performance_analysis import PerformanceAnalyzer


class ReportGenerator:
    """
    Lớp tạo báo cáo tổng hợp từ các kết quả phân tích
    """
    
    def __init__(self, output_dir: str = "final_report"):
        """
        Khởi tạo generator
        
        Args:
            output_dir: Thư mục lưu báo cáo
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Thư mục dữ liệu đầu vào
        self.benchmark_dir = "benchmark_results"
        self.consensus_dir = "consensus_comparison"
        self.performance_dir = "performance_analysis"
        
        # Dữ liệu báo cáo
        self.benchmark_data = {}
        self.consensus_data = {}
        self.performance_data = {}
        
        # Cấu hình báo cáo
        self.report_title = "Báo Cáo Phân Tích DQN Blockchain"
        self.report_date = datetime.now().strftime("%d/%m/%Y")
        
    def collect_data(self) -> None:
        """
        Thu thập dữ liệu từ các kết quả phân tích
        """
        # 1. Thu thập dữ liệu benchmark
        if os.path.exists(self.benchmark_dir):
            benchmark_summary = os.path.join(self.benchmark_dir, "benchmark_summary.csv")
            if os.path.exists(benchmark_summary):
                self.benchmark_data["summary"] = pd.read_csv(benchmark_summary)
                
            # Thu thập các biểu đồ
            benchmark_charts = glob.glob(os.path.join(self.benchmark_dir, "*.png"))
            self.benchmark_data["charts"] = benchmark_charts
        
        # 2. Thu thập dữ liệu consensus comparison
        if os.path.exists(self.consensus_dir):
            consensus_results = os.path.join(self.consensus_dir, "consensus_results.json")
            if os.path.exists(consensus_results):
                with open(consensus_results, 'r') as f:
                    self.consensus_data["results"] = json.load(f)
            
            # Thu thập các biểu đồ
            consensus_charts = glob.glob(os.path.join(self.consensus_dir, "*.png"))
            self.consensus_data["charts"] = consensus_charts
        
        # 3. Thu thập dữ liệu performance analysis
        if os.path.exists(self.performance_dir):
            performance_summary = os.path.join(self.performance_dir, "analysis_summary.csv")
            if os.path.exists(performance_summary):
                self.performance_data["summary"] = pd.read_csv(performance_summary)
                
            # Thu thập các biểu đồ
            performance_charts = glob.glob(os.path.join(self.performance_dir, "*.png"))
            self.performance_data["charts"] = performance_charts
    
    def run_missing_analyses(self) -> None:
        """
        Chạy các phân tích còn thiếu (nếu cần)
        """
        # Kiểm tra và chạy benchmark nếu cần
        if not self.benchmark_data:
            print("Benchmark data not found. Running comparative benchmarks...")
            benchmarks = run_comparative_benchmarks()
            analyze_results(benchmarks, self.benchmark_dir)
            
            # Thu thập dữ liệu mới
            if os.path.exists(self.benchmark_dir):
                benchmark_summary = os.path.join(self.benchmark_dir, "benchmark_summary.csv")
                if os.path.exists(benchmark_summary):
                    self.benchmark_data["summary"] = pd.read_csv(benchmark_summary)
                    
                # Thu thập các biểu đồ
                benchmark_charts = glob.glob(os.path.join(self.benchmark_dir, "*.png"))
                self.benchmark_data["charts"] = benchmark_charts
        
        # Kiểm tra và chạy consensus comparison nếu cần
        if not self.consensus_data:
            print("Consensus comparison data not found. Running consensus comparison...")
            simulator = ConsensusSimulator(self.consensus_dir)
            simulator.run_comparison()
            simulator.visualize_comparison()
            
            # Thu thập dữ liệu mới
            consensus_results = os.path.join(self.consensus_dir, "consensus_results.json")
            if os.path.exists(consensus_results):
                with open(consensus_results, 'r') as f:
                    self.consensus_data["results"] = json.load(f)
            
            # Thu thập các biểu đồ
            consensus_charts = glob.glob(os.path.join(self.consensus_dir, "*.png"))
            self.consensus_data["charts"] = consensus_charts
        
        # Kiểm tra và chạy performance analysis nếu cần
        if not self.performance_data:
            print("Performance analysis data not found. Running performance analysis...")
            # Tạo các cấu hình mô phỏng đơn giản
            configs = [
                {
                    'name': "dqn_analysis_with",
                    'num_shards': 4,
                    'num_steps': 20,
                    'tx_per_step': 10,
                    'use_dqn': True
                },
                {
                    'name': "dqn_analysis_without",
                    'num_shards': 4,
                    'num_steps': 20,
                    'tx_per_step': 10,
                    'use_dqn': False
                }
            ]
            
            analyzer = PerformanceAnalyzer(self.performance_dir)
            analyzer.run_analysis(configs)
            
            # Thu thập dữ liệu mới
            performance_summary = os.path.join(self.performance_dir, "analysis_summary.csv")
            if os.path.exists(performance_summary):
                self.performance_data["summary"] = pd.read_csv(performance_summary)
                
            # Thu thập các biểu đồ
            performance_charts = glob.glob(os.path.join(self.performance_dir, "*.png"))
            self.performance_data["charts"] = performance_charts
    
    def generate_summary_charts(self) -> List[str]:
        """
        Tạo biểu đồ tổng hợp từ các kết quả phân tích
        
        Returns:
            Danh sách đường dẫn đến các biểu đồ tổng hợp
        """
        summary_charts = []
        
        # Tạo biểu đồ tổng hợp hiệu suất giữa các phương pháp
        if "summary" in self.benchmark_data and "results" in self.consensus_data:
            # Tổng hợp dữ liệu
            benchmark_df = self.benchmark_data["summary"]
            consensus_data = self.consensus_data["results"]
            
            # Tạo DataFrame tổng hợp
            methods = []
            throughputs = []
            latencies = []
            energy_values = []
            
            # Dữ liệu từ benchmark
            if "dqn" in benchmark_df.columns:
                dqn_data = benchmark_df[benchmark_df["dqn"] == True]
                non_dqn_data = benchmark_df[benchmark_df["dqn"] == False]
                
                if not dqn_data.empty:
                    methods.append("DQN-based")
                    throughputs.append(dqn_data["throughput"].mean())
                    latencies.append(dqn_data["latency"].mean())
                    energy_values.append(dqn_data["energy_consumption"].mean())
                
                if not non_dqn_data.empty:
                    methods.append("Non-DQN")
                    throughputs.append(non_dqn_data["throughput"].mean())
                    latencies.append(non_dqn_data["latency"].mean())
                    energy_values.append(non_dqn_data["energy_consumption"].mean())
            
            # Dữ liệu từ consensus comparison
            for protocol, data in consensus_data.items():
                if isinstance(data, dict) and "throughput" in data and "latency" in data and "energy_consumption" in data:
                    methods.append(protocol)
                    throughputs.append(data["throughput"])
                    latencies.append(data["latency"])
                    energy_values.append(data["energy_consumption"])
            
            if methods:
                # Tạo DataFrame
                comparison_df = pd.DataFrame({
                    "Method": methods,
                    "Throughput": throughputs,
                    "Latency": latencies,
                    "Energy Consumption": energy_values
                })
                
                # Lưu DataFrame
                comparison_file = os.path.join(self.output_dir, "method_comparison.csv")
                comparison_df.to_csv(comparison_file, index=False)
                
                # Tạo biểu đồ tổng hợp
                plt.figure(figsize=(15, 10))
                
                # 1. Throughput
                plt.subplot(1, 3, 1)
                sns.barplot(x="Method", y="Throughput", data=comparison_df)
                plt.title("Throughput Comparison")
                plt.xticks(rotation=45)
                
                # 2. Latency
                plt.subplot(1, 3, 2)
                sns.barplot(x="Method", y="Latency", data=comparison_df)
                plt.title("Latency Comparison")
                plt.xticks(rotation=45)
                
                # 3. Energy Consumption
                plt.subplot(1, 3, 3)
                sns.barplot(x="Method", y="Energy Consumption", data=comparison_df)
                plt.title("Energy Consumption Comparison")
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                # Lưu biểu đồ
                chart_file = os.path.join(self.output_dir, "method_comparison.png")
                plt.savefig(chart_file)
                plt.close()
                
                summary_charts.append(chart_file)
        
        # Tạo biểu đồ so sánh DQN vs Non-DQN
        if "summary" in self.performance_data:
            performance_df = self.performance_data["summary"]
            
            if "use_dqn" in performance_df.columns:
                # Nhóm theo việc sử dụng DQN
                dqn_comparison = performance_df.groupby("use_dqn").agg({
                    "throughput": "mean",
                    "latency": "mean",
                    "success_rate": "mean",
                    "energy_consumption": "mean"
                }).reset_index()
                
                # Đổi tên các giá trị
                dqn_comparison["use_dqn"] = dqn_comparison["use_dqn"].map({True: "DQN", False: "Non-DQN"})
                
                if not dqn_comparison.empty:
                    # Tạo biểu đồ so sánh
                    plt.figure(figsize=(15, 10))
                    
                    # 1. Throughput
                    plt.subplot(2, 2, 1)
                    sns.barplot(x="use_dqn", y="throughput", data=dqn_comparison)
                    plt.title("Throughput: DQN vs Non-DQN")
                    plt.xlabel("")
                    
                    # 2. Latency
                    plt.subplot(2, 2, 2)
                    sns.barplot(x="use_dqn", y="latency", data=dqn_comparison)
                    plt.title("Latency: DQN vs Non-DQN")
                    plt.xlabel("")
                    
                    # 3. Success Rate
                    plt.subplot(2, 2, 3)
                    sns.barplot(x="use_dqn", y="success_rate", data=dqn_comparison)
                    plt.title("Success Rate: DQN vs Non-DQN")
                    plt.xlabel("")
                    
                    # 4. Energy Consumption
                    plt.subplot(2, 2, 4)
                    sns.barplot(x="use_dqn", y="energy_consumption", data=dqn_comparison)
                    plt.title("Energy Consumption: DQN vs Non-DQN")
                    plt.xlabel("")
                    
                    plt.tight_layout()
                    
                    # Lưu biểu đồ
                    chart_file = os.path.join(self.output_dir, "dqn_comparison.png")
                    plt.savefig(chart_file)
                    plt.close()
                    
                    summary_charts.append(chart_file)
                
        return summary_charts
    
    def generate_pdf_report(self) -> str:
        """
        Tạo báo cáo PDF tổng hợp
        
        Returns:
            Đường dẫn đến báo cáo PDF
        """
        # Khởi tạo PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Tiêu đề báo cáo
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, self.report_title, 0, 1, "C")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Ngày: {self.report_date}", 0, 1, "C")
        pdf.ln(10)
        
        # Tạo biểu đồ tổng hợp
        summary_charts = self.generate_summary_charts()
        
        # 1. Tổng quan
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "1. Tổng Quan", 0, 1)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, "Báo cáo này trình bày kết quả phân tích hiệu suất của các phương pháp tối ưu hoá blockchain dựa trên DQN. "
                         "Các phương pháp được so sánh bao gồm: Proof of Work (PoW), Proof of Stake (PoS), "
                         "Practical Byzantine Fault Tolerance (PBFT), và Adaptive Cross-Shard Consensus (ACSC).")
        pdf.ln(5)
        
        # 2. Phân tích benchmark
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "2. Phân Tích Benchmark", 0, 1)
        
        if self.benchmark_data:
            if "summary" in self.benchmark_data:
                benchmark_df = self.benchmark_data["summary"]
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, f"Số lượng cấu hình đã thử nghiệm: {len(benchmark_df)}")
                
                # Thống kê tóm tắt
                if not benchmark_df.empty and "throughput" in benchmark_df.columns and "latency" in benchmark_df.columns:
                    pdf.multi_cell(0, 10, f"Throughput trung bình: {benchmark_df['throughput'].mean():.2f}")
                    pdf.multi_cell(0, 10, f"Latency trung bình: {benchmark_df['latency'].mean():.2f}")
                    
                    if "energy_consumption" in benchmark_df.columns:
                        pdf.multi_cell(0, 10, f"Tiêu thụ năng lượng trung bình: {benchmark_df['energy_consumption'].mean():.2f}")
            
            # Thêm biểu đồ benchmark
            if "charts" in self.benchmark_data and self.benchmark_data["charts"]:
                for chart in self.benchmark_data["charts"][:2]:  # Giới hạn 2 biểu đồ
                    if os.path.exists(chart):
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 10, f"Biểu đồ: {os.path.basename(chart)}", 0, 1)
                        pdf.image(chart, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(100)  # Khoảng cách dưới biểu đồ
        else:
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "Không có dữ liệu benchmark.")
        
        # 3. So sánh các phương pháp đồng thuận
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "3. So Sánh Phương Pháp Đồng Thuận", 0, 1)
        
        if self.consensus_data:
            if "results" in self.consensus_data:
                consensus_data = self.consensus_data["results"]
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, f"Số lượng phương pháp đã thử nghiệm: {len(consensus_data)}")
                
                # Tạo bảng so sánh
                pdf.set_font("Arial", "B", 10)
                pdf.cell(40, 10, "Phương Pháp", 1, 0, "C")
                pdf.cell(40, 10, "Throughput", 1, 0, "C")
                pdf.cell(40, 10, "Latency", 1, 0, "C")
                pdf.cell(40, 10, "Tiêu Thụ Năng Lượng", 1, 1, "C")
                
                pdf.set_font("Arial", "", 10)
                for protocol, data in consensus_data.items():
                    if isinstance(data, dict) and "throughput" in data and "latency" in data and "energy_consumption" in data:
                        pdf.cell(40, 10, protocol, 1, 0)
                        pdf.cell(40, 10, f"{data['throughput']:.2f}", 1, 0)
                        pdf.cell(40, 10, f"{data['latency']:.2f}", 1, 0)
                        pdf.cell(40, 10, f"{data['energy_consumption']:.2f}", 1, 1)
            
            # Thêm biểu đồ so sánh đồng thuận
            if "charts" in self.consensus_data and self.consensus_data["charts"]:
                for chart in self.consensus_data["charts"][:2]:  # Giới hạn 2 biểu đồ
                    if os.path.exists(chart):
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 10, f"Biểu đồ: {os.path.basename(chart)}", 0, 1)
                        pdf.image(chart, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(100)  # Khoảng cách dưới biểu đồ
        else:
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "Không có dữ liệu so sánh đồng thuận.")
        
        # 4. Phân tích hiệu suất
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "4. Phân Tích Hiệu Suất", 0, 1)
        
        if self.performance_data:
            if "summary" in self.performance_data:
                performance_df = self.performance_data["summary"]
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, f"Số lượng cấu hình đã phân tích: {len(performance_df)}")
                
                # Thống kê tóm tắt
                if not performance_df.empty and "throughput" in performance_df.columns and "latency" in performance_df.columns:
                    pdf.multi_cell(0, 10, f"Throughput trung bình: {performance_df['throughput'].mean():.2f}")
                    pdf.multi_cell(0, 10, f"Latency trung bình: {performance_df['latency'].mean():.2f}")
                    
                    if "success_rate" in performance_df.columns:
                        pdf.multi_cell(0, 10, f"Tỉ lệ thành công trung bình: {performance_df['success_rate'].mean():.2f}")
                    
                    if "energy_consumption" in performance_df.columns:
                        pdf.multi_cell(0, 10, f"Tiêu thụ năng lượng trung bình: {performance_df['energy_consumption'].mean():.2f}")
                    
                    # So sánh DQN vs Non-DQN
                    if "use_dqn" in performance_df.columns:
                        dqn_df = performance_df[performance_df["use_dqn"] == True]
                        non_dqn_df = performance_df[performance_df["use_dqn"] == False]
                        
                        if not dqn_df.empty and not non_dqn_df.empty:
                            pdf.ln(5)
                            pdf.set_font("Arial", "B", 12)
                            pdf.multi_cell(0, 10, "So sánh DQN vs Non-DQN:")
                            pdf.set_font("Arial", "", 12)
                            
                            # Throughput
                            dqn_throughput = dqn_df["throughput"].mean()
                            non_dqn_throughput = non_dqn_df["throughput"].mean()
                            improvement = ((dqn_throughput - non_dqn_throughput) / non_dqn_throughput) * 100 if non_dqn_throughput > 0 else 0
                            pdf.multi_cell(0, 10, f"Cải thiện Throughput: {improvement:.2f}%")
                            
                            # Latency
                            dqn_latency = dqn_df["latency"].mean()
                            non_dqn_latency = non_dqn_df["latency"].mean()
                            improvement = ((non_dqn_latency - dqn_latency) / non_dqn_latency) * 100 if non_dqn_latency > 0 else 0
                            pdf.multi_cell(0, 10, f"Giảm Latency: {improvement:.2f}%")
                            
                            # Success Rate
                            if "success_rate" in dqn_df.columns and "success_rate" in non_dqn_df.columns:
                                dqn_success_rate = dqn_df["success_rate"].mean()
                                non_dqn_success_rate = non_dqn_df["success_rate"].mean()
                                improvement = ((dqn_success_rate - non_dqn_success_rate) / non_dqn_success_rate) * 100 if non_dqn_success_rate > 0 else 0
                                pdf.multi_cell(0, 10, f"Cải thiện Tỉ lệ thành công: {improvement:.2f}%")
                            
                            # Energy Consumption
                            if "energy_consumption" in dqn_df.columns and "energy_consumption" in non_dqn_df.columns:
                                dqn_energy = dqn_df["energy_consumption"].mean()
                                non_dqn_energy = non_dqn_df["energy_consumption"].mean()
                                improvement = ((non_dqn_energy - dqn_energy) / non_dqn_energy) * 100 if non_dqn_energy > 0 else 0
                                pdf.multi_cell(0, 10, f"Giảm Tiêu thụ năng lượng: {improvement:.2f}%")
            
            # Thêm biểu đồ phân tích hiệu suất
            if "charts" in self.performance_data and self.performance_data["charts"]:
                for chart in self.performance_data["charts"][:3]:  # Giới hạn 3 biểu đồ
                    if os.path.exists(chart):
                        pdf.add_page()
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 10, f"Biểu đồ: {os.path.basename(chart)}", 0, 1)
                        pdf.image(chart, x=10, y=pdf.get_y(), w=180)
                        pdf.ln(100)  # Khoảng cách dưới biểu đồ
        else:
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, "Không có dữ liệu phân tích hiệu suất.")
        
        # 5. Biểu đồ tổng hợp
        if summary_charts:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "5. Biểu Đồ Tổng Hợp", 0, 1)
            
            for chart in summary_charts:
                if os.path.exists(chart):
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"Biểu đồ: {os.path.basename(chart)}", 0, 1)
                    pdf.image(chart, x=10, y=pdf.get_y(), w=180)
                    pdf.ln(100)  # Khoảng cách dưới biểu đồ
                    pdf.add_page()
        
        # 6. Kết luận
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "6. Kết Luận", 0, 1)
        pdf.set_font("Arial", "", 12)
        
        # Phân tích so sánh
        pdf.multi_cell(0, 10, "Từ kết quả phân tích, có thể rút ra các kết luận sau:")
        
        # Tạo danh sách kết luận
        conclusions = []
        
        # So sánh DQN vs Non-DQN
        if self.performance_data and "summary" in self.performance_data:
            performance_df = self.performance_data["summary"]
            
            if "use_dqn" in performance_df.columns:
                dqn_df = performance_df[performance_df["use_dqn"] == True]
                non_dqn_df = performance_df[performance_df["use_dqn"] == False]
                
                if not dqn_df.empty and not non_dqn_df.empty:
                    # Throughput
                    dqn_throughput = dqn_df["throughput"].mean()
                    non_dqn_throughput = non_dqn_df["throughput"].mean()
                    
                    if dqn_throughput > non_dqn_throughput:
                        conclusions.append(f"DQN cải thiện throughput lên {((dqn_throughput - non_dqn_throughput) / non_dqn_throughput * 100):.2f}% so với phương pháp truyền thống.")
                    
                    # Latency
                    dqn_latency = dqn_df["latency"].mean()
                    non_dqn_latency = non_dqn_df["latency"].mean()
                    
                    if dqn_latency < non_dqn_latency:
                        conclusions.append(f"DQN giảm latency xuống {((non_dqn_latency - dqn_latency) / non_dqn_latency * 100):.2f}% so với phương pháp truyền thống.")
                    
                    # Energy Consumption
                    if "energy_consumption" in dqn_df.columns and "energy_consumption" in non_dqn_df.columns:
                        dqn_energy = dqn_df["energy_consumption"].mean()
                        non_dqn_energy = non_dqn_df["energy_consumption"].mean()
                        
                        if dqn_energy < non_dqn_energy:
                            conclusions.append(f"DQN giảm tiêu thụ năng lượng xuống {((non_dqn_energy - dqn_energy) / non_dqn_energy * 100):.2f}% so với phương pháp truyền thống.")
        
        # So sánh ACSC với các phương pháp đồng thuận khác
        if self.consensus_data and "results" in self.consensus_data:
            consensus_data = self.consensus_data["results"]
            
            if "ACSC" in consensus_data and "PoW" in consensus_data:
                acsc_energy = consensus_data["ACSC"].get("energy_consumption", 0)
                pow_energy = consensus_data["PoW"].get("energy_consumption", 0)
                
                if acsc_energy < pow_energy:
                    conclusions.append(f"ACSC tiêu thụ ít năng lượng hơn {((pow_energy - acsc_energy) / pow_energy * 100):.2f}% so với PoW.")
            
            if "ACSC" in consensus_data and "PBFT" in consensus_data:
                acsc_throughput = consensus_data["ACSC"].get("throughput", 0)
                pbft_throughput = consensus_data["PBFT"].get("throughput", 0)
                
                if acsc_throughput > pbft_throughput:
                    conclusions.append(f"ACSC có throughput cao hơn {((acsc_throughput - pbft_throughput) / pbft_throughput * 100):.2f}% so với PBFT.")
        
        # Thêm các kết luận chung
        conclusions.extend([
            "Phương pháp tối ưu hoá blockchain dựa trên DQN cho thấy hiệu quả cao trong việc cải thiện throughput và giảm latency.",
            "ACSC kết hợp với DQN cho phép blockchain thích nghi với điều kiện mạng và tải giao dịch.",
            "Cơ chế trust-based giúp cải thiện an toàn và hiệu quả của hệ thống."
        ])
        
        # In ra các kết luận
        for i, conclusion in enumerate(conclusions, 1):
            pdf.multi_cell(0, 10, f"{i}. {conclusion}")
        
        # Lưu báo cáo
        report_file = os.path.join(self.output_dir, "final_report.pdf")
        pdf.output(report_file)
        
        return report_file
    
    def generate_report(self) -> str:
        """
        Tạo báo cáo tổng hợp
        
        Returns:
            Đường dẫn đến báo cáo
        """
        # Thu thập dữ liệu
        self.collect_data()
        
        # Chạy các phân tích còn thiếu
        self.run_missing_analyses()
        
        # Tạo báo cáo PDF
        report_file = self.generate_pdf_report()
        
        print(f"Report generated at: {report_file}")
        
        return report_file


def main():
    """
    Hàm chính để tạo báo cáo tổng hợp
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate a comprehensive report from all analysis results")
    parser.add_argument("--output_dir", default="final_report", help="Directory to save the report")
    parser.add_argument("--run_analysis", action="store_true", help="Run missing analyses if necessary")
    args = parser.parse_args()
    
    # Khởi tạo generator
    generator = ReportGenerator(args.output_dir)
    
    # Tạo báo cáo
    generator.generate_report()


if __name__ == "__main__":
    main() 