#!/usr/bin/env python
"""
Script to run advanced blockchain simulation with improvements
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from dqn_blockchain_sim.simulation.advanced_simulation import AdvancedSimulation
import logging


def main():
    """
    Main function to launch advanced simulation
    """
    parser = argparse.ArgumentParser(
        description='Run advanced blockchain simulation with all improvements'
    )

    # Simulation setup
    parser.add_argument('--num_shards', type=int, default=8,
                        help='Number of shards in the network (default: 8)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of simulation steps (default: 1000)')
    parser.add_argument('--tx_per_step', type=int, default=50,
                        help='Number of transactions per step (default: 50)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of simulation iterations for statistical accuracy (default: 5)')
    
    # Module options
    parser.add_argument('--use_real_data', action='store_true',
                        help='Use real Ethereum data')
    parser.add_argument('--use_dqn', action='store_true', default=True,
                        help='Use DQN agent (default: True)')
    parser.add_argument('--eth_api_key', type=str,
                        help='Ethereum API key (required if --use_real_data)')
    parser.add_argument('--network_latency', type=int, default=100,
                        help='Base network latency in ms (default: 100)')
    parser.add_argument('--latency_variation', type=float, default=0.2,
                        help='Network latency variation factor (default: 0.2)')
    parser.add_argument('--packet_loss', type=float, default=0.01,
                        help='Packet loss probability (default: 0.01)')
    parser.add_argument('--block_time', type=float, default=2.0,
                        help='Block creation time in seconds (default: 2.0)')

    # Output options
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory (default: "data")')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory (default: "logs")')
    parser.add_argument('--visualize', action='store_true',
                        help='Display result charts')
    parser.add_argument('--save_stats', action='store_true', default=True,
                        help='Save statistics to file (default: True)')
    parser.add_argument('--run_id', type=str, 
                        default=str(int(time.time())),
                        help='Unique run identifier (default: current timestamp)')

    # Parse arguments
    args = parser.parse_args()

    # Check parameters
    if args.use_real_data and not args.eth_api_key:
        parser.error("--use_real_data requires --eth_api_key")
    
    # Create output directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Display simulation information
    print(f"Starting simulation with {args.num_shards} shards, running for {args.steps} steps")
    print(f"Transactions per step: {args.tx_per_step}")
    print(f"Number of iterations: {args.iterations}")
    print(f"Using real data: {args.use_real_data}")
    print(f"Using DQN agent: {args.use_dqn}")
    print(f"Network parameters: Latency={args.network_latency}ms, Variation={args.latency_variation*100}%, Packet loss={args.packet_loss*100}%")
    print(f"Block time: {args.block_time} seconds")
    print(f"Run ID: {args.run_id}")
    
    # Store simulation configuration
    config = vars(args)
    config_path = os.path.join(args.data_dir, f"config_{args.run_id}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")
    
    # Collect results from all iterations
    all_results = []
    
    for iteration in range(args.iterations):
        print(f"\n=== Running iteration {iteration+1}/{args.iterations} ===")
        
        # Initialize simulation with network parameters
        simulation = AdvancedSimulation(
            num_shards=args.num_shards,
            use_real_data=args.use_real_data,
            use_dqn=args.use_dqn,
            eth_api_key=args.eth_api_key,
            data_dir=args.data_dir,
            log_dir=args.log_dir
        )

        # Cập nhật cấu hình mạng
        if hasattr(simulation, 'network') and simulation.network:
            # Cập nhật các thuộc tính mạng nếu có
            if hasattr(args, 'network_latency'):
                simulation.network.base_latency = args.network_latency
            if hasattr(args, 'latency_variation'):
                simulation.network.latency_variation = args.latency_variation
            if hasattr(args, 'packet_loss'):
                simulation.network.packet_loss_prob = args.packet_loss
            if hasattr(args, 'block_time'):
                simulation.network.block_time = args.block_time
        
        # Run simulation
        results = simulation.run_simulation(
            num_steps=args.steps,
            tx_per_step=args.tx_per_step,
            visualize=False,  # Don't visualize individual iterations
            save_stats=False  # Don't save individual iterations
        )
        
        # In ra kết quả chi tiết để debug
        print("\n=== DEBUG: Simulation Results ===")
        print(f"Type of results: {type(results)}")
        print(f"Keys in results: {list(results.keys()) if isinstance(results, dict) else 'Not a dictionary'}")
        
        if isinstance(results, dict):
            if 'transaction_stats' in results:
                print("\nTransaction Stats:")
                for k, v in results['transaction_stats'].items():
                    print(f"  {k}: {v}")
            
            if 'performance_stats' in results:
                print("\nPerformance Stats:")
                for k, v in results['performance_stats'].items():
                    print(f"  {k}: {v}")
                    
            if 'module_stats' in results:
                print("\nModule Stats:")
                for module, stats in results['module_stats'].items():
                    print(f"  {module}:")
                    for k, v in stats.items():
                        print(f"    {k}: {v}")
                        
            # Kiểm tra các trường khác
            for key in results.keys():
                if key not in ['transaction_stats', 'performance_stats', 'module_stats', 'iteration']:
                    print(f"\nOther field: {key}")
                    if isinstance(results[key], dict):
                        for k, v in results[key].items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"  Value: {results[key]}")
        
            # Lưu thông tin DEBUG STATS nếu có
            debug_stats = None
            for line in str(results).split('\n'):
                if 'DEBUG STATS:' in line:
                    try:
                        # Trích xuất chuỗi DEBUG STATS
                        debug_str = line.split('DEBUG STATS:', 1)[1].strip()
                        # Chuyển đổi chuỗi thành dictionary
                        import ast
                        debug_stats = ast.literal_eval(debug_str)
                        
                        # Thêm thông tin từ DEBUG STATS vào module_stats
                        if 'module_stats' not in results:
                            results['module_stats'] = {}
                        
                        if 'mad_rapid' not in results['module_stats']:
                            results['module_stats']['mad_rapid'] = {}
                        
                        # Lấy thông tin từ cross_shard_tx_stats
                        if 'cross_shard_tx_stats' in debug_stats:
                            cs_stats = debug_stats['cross_shard_tx_stats']
                            results['module_stats']['mad_rapid']['total_tx_processed'] = cs_stats.get('total_attempts', 0)
                            results['module_stats']['mad_rapid']['energy_saved'] = cs_stats.get('energy_saved', 0)
                        
                        # Lấy thông tin từ path_selection_stats
                        if 'path_selection_stats' in debug_stats:
                            path_stats = debug_stats['path_selection_stats']
                            results['module_stats']['mad_rapid']['optimized_tx_count'] = path_stats.get('successful_optimizations', 0)
                            
                            # Nếu có total_energy_saved, sử dụng giá trị lớn hơn
                            if 'total_energy_saved' in path_stats:
                                current_energy = results['module_stats']['mad_rapid'].get('energy_saved', 0)
                                path_energy = path_stats.get('total_energy_saved', 0)
                                results['module_stats']['mad_rapid']['energy_saved'] = max(current_energy, path_energy)
                    except Exception as e:
                        print(f"Lỗi khi xử lý DEBUG STATS: {e}")
        
        # Store iteration results
        results["iteration"] = iteration
        all_results.append(results)
        
        # Show condensed iteration results
        print(f"\n--- Iteration {iteration+1} Results ---")
        if isinstance(results, dict) and 'transaction_stats' in results:
            tx_stats = results['transaction_stats']
            print(f"Success rate: {tx_stats.get('success_rate', 0):.2%}")
            print(f"Cross-shard success rate: {tx_stats.get('cross_shard_success_rate', 0):.2%}")
        else:
            print("Warning: Could not find transaction_stats in results")
            
        if isinstance(results, dict) and 'performance_metrics' in results:
            print(f"Congestion level: {results['performance_metrics'].get('avg_congestion', 0):.2%}")

    # Calculate aggregate statistics
    aggregate_results = calculate_aggregate_statistics(all_results)
    
    # In ra kết quả tổng hợp để debug
    print("\n=== DEBUG: Aggregate Results ===")
    print(f"Type of aggregate_results: {type(aggregate_results)}")
    print(f"Keys in aggregate_results: {list(aggregate_results.keys()) if isinstance(aggregate_results, dict) else 'Not a dictionary'}")
    
    if isinstance(aggregate_results, dict):
        if 'transaction_stats' in aggregate_results:
            print("\nAggregate Transaction Stats:")
            for k, v in aggregate_results['transaction_stats'].items():
                print(f"  {k}: {v}")
    
    aggregate_results["run_id"] = args.run_id
    aggregate_results["config"] = config
    aggregate_results["timestamp"] = datetime.now().isoformat()
    
    # Save aggregate results
    if args.save_stats:
        save_path = os.path.join(args.data_dir, f"results_{args.run_id}.json")
        with open(save_path, 'w') as f:
            json.dump(aggregate_results, f, indent=2)
        print(f"\nAggregate results saved to {save_path}")
    
    # Display summary results
    print("\n===== SIMULATION AGGREGATE RESULTS =====")
    print(f"Total transactions processed: {aggregate_results['transaction_stats']['total_transactions']}")
    print(f"Successful transactions: {aggregate_results['transaction_stats']['successful_transactions']}")
    print(f"Overall success rate: {aggregate_results['transaction_stats']['success_rate']:.2%}")
    print(f"Cross-shard success rate: {aggregate_results['transaction_stats'].get('cross_shard_success_rate', 0):.2%}")
    print(f"95% confidence interval: {aggregate_results['transaction_stats']['success_rate_95ci']}")

    if 'performance_metrics' in aggregate_results:
        metrics = aggregate_results['performance_metrics']
        print("\n--- AVERAGE PERFORMANCE ---")
        print(f"Throughput: {metrics.get('avg_throughput', 0):.2f} transactions/step")
        print(f"Latency: {metrics.get('avg_latency', 0):.2f} ms")
        print(f"Congestion level: {metrics.get('congestion_level', 0):.2%}")
        print(f"Energy consumption: {metrics.get('energy_consumption', 0):.2f} units")

    print("\n=== MODULE STATISTICS ===")

    # MAD-RAPID
    if 'module_stats' in aggregate_results and 'mad_rapid' in aggregate_results['module_stats']:
        mad_rapid_stats = aggregate_results['module_stats']['mad_rapid']
        # Chuyển đổi các khóa sang định dạng cũ cho việc hiển thị
        aggregate_results['module_stats']['mad_rapid']['total_cross_shard_transactions'] = mad_rapid_stats.get('total_tx', 0)
        aggregate_results['module_stats']['mad_rapid']['optimized_transactions'] = mad_rapid_stats.get('optimized_tx', 0)
        aggregate_results['module_stats']['mad_rapid']['latency_improvement_percent'] = mad_rapid_stats.get('latency_improvement', 0) * 100 if isinstance(mad_rapid_stats.get('latency_improvement', 0), (int, float)) else 0
        
        mad_rapid_stats = aggregate_results['module_stats']['mad_rapid']
        print("\nMAD-RAPID Protocol:")
        print(f"- Total cross-shard transactions: {mad_rapid_stats.get('total_cross_shard_transactions', 0)}")
        print(f"- Optimized transactions: {mad_rapid_stats.get('optimized_transactions', 0)}")
        latency_imp = mad_rapid_stats.get('latency_improvement_percent', 0)
        print(f"- Latency improvement: {latency_imp:.2f}%")
        print(f"- Energy saved: {mad_rapid_stats.get('energy_saved', 0):.2f} units")
    
    # HTDCM
    if 'module_stats' in aggregate_results and 'htdcm' in aggregate_results['module_stats']:
        htdcm_stats = aggregate_results['module_stats']['htdcm']
        print("\nHTDCM:")
        print(f"- Average trust score: {htdcm_stats.get('avg_trust_score', 0):.2f}")
        print(f"- Trust violations detected: {htdcm_stats.get('trust_violations_detected', 0)}")
        tx_security = htdcm_stats.get('transaction_security_level', {})
        print(f"- High security transactions: {tx_security.get('high', 0)}")
        print(f"- Medium security transactions: {tx_security.get('medium', 0)}")
        print(f"- Low security transactions: {tx_security.get('low', 0)}")

    # ACSC
    if 'module_stats' in aggregate_results and 'acsc' in aggregate_results['module_stats']:
        acsc_stats = aggregate_results['module_stats']['acsc']
        print("\nACSC:")
        print(f"- Total transactions: {acsc_stats.get('total_transactions', 0)}")
        print(f"- Fast Consensus usage: {acsc_stats.get('fast_consensus_percent', 0):.2f}%")
        print(f"- Standard Consensus usage: {acsc_stats.get('standard_consensus_percent', 0):.2f}%")
        print(f"- Robust Consensus usage: {acsc_stats.get('robust_consensus_percent', 0):.2f}%")
        print(f"- Consensus success rate: {acsc_stats.get('consensus_success_rate', 0):.2f}%")
    
    # Visualize aggregate results if requested
    if args.visualize:
        try:
            # Import visualization only when needed
            from dqn_blockchain_sim.utils.visualizer import SimulationVisualizer
            visualizer = SimulationVisualizer()
            visualizer.visualize_aggregate_results(aggregate_results)
        except Exception as e:
            print(f"Error during visualization: {e}")

    return 0


def calculate_aggregate_statistics(results_list):
    """
    Tính toán các thống kê tổng hợp từ danh sách kết quả mô phỏng
    
    Args:
        results_list: Danh sách kết quả từ các lần lặp
        
    Returns:
        Dict chứa các thống kê tổng hợp
    """
    import numpy as np
    from scipy import stats
    
    # Nếu không có kết quả, trả về từ điển trống
    if not results_list:
        return {}
    
    # Khởi tạo kết quả tổng hợp với cấu trúc tương tự kết quả đơn lẻ
    aggregate_results = {
        "transaction_stats": {
            "total_transactions": 0,
            "successful_transactions": 0,
            "success_rate": 0,
            "cross_shard_transactions": 0,
            "successful_cross_shard_transactions": 0,
            "cross_shard_success_rate": 0
        },
        "performance_stats": {
            "avg_throughput": 0,
            "avg_latency": 0,
            "energy_consumption": 0,
            "congestion_level": 0
        },
        "module_stats": {
            "mad_rapid": {
                "total_tx": 0,
                "optimized_tx": 0,
                "latency_improvement": 0,
                "energy_saved": 0
            },
            "acsc": {
                "total_tx": 0,
                "fast_consensus": 0,
                "standard_consensus": 0,
                "robust_consensus": 0,
                "consensus_success_rate": 0
            }
        }
    }
    
    # Khởi tạo các danh sách để tính trung bình và độ lệch chuẩn
    success_rates = []
    cross_shard_success_rates = []
    throughputs = []
    latencies = []
    energy_consumptions = []
    congestion_levels = []
    
    # Các biến tổng hợp cho module MAD-RAPID và ACSC
    mad_rapid_total_tx = 0
    mad_rapid_optimized_tx = 0
    mad_rapid_latency_improvements = 0
    mad_rapid_energy_saved = 0
    
    acsc_total_tx = 0
    acsc_fast_consensus = 0
    acsc_standard_consensus = 0
    acsc_robust_consensus = 0
    
    # Xử lý từng kết quả
    for i, results in enumerate(results_list):
        # Extract transaction statistics
        tx_stats = results.get("transaction_stats", {})
        success_rate = tx_stats.get("success_rate", 0)
        success_rates.append(success_rate)
        
        # Extract cross-shard transaction statistics
        cross_shard_tx = tx_stats.get("cross_shard_transactions", {})
        cross_shard_success_rate = cross_shard_tx.get("success_rate", 0)
        cross_shard_success_rates.append(cross_shard_success_rate)
        
        # Update total transaction counts
        aggregate_results["transaction_stats"]["total_transactions"] += tx_stats.get("total", 0)
        aggregate_results["transaction_stats"]["successful_transactions"] += tx_stats.get("successful", 0)
        aggregate_results["transaction_stats"]["cross_shard_transactions"] += cross_shard_tx.get("total", 0)
        aggregate_results["transaction_stats"]["successful_cross_shard_transactions"] += cross_shard_tx.get("successful", 0)
        
        # Extract performance metrics
        perf_stats = results.get("performance_stats", {})
        throughputs.append(perf_stats.get("avg_throughput", 0))
        latencies.append(perf_stats.get("avg_latency", 0))
        energy_consumptions.append(perf_stats.get("energy_consumption", 0))
        congestion_levels.append(perf_stats.get("congestion_level", 0))
        
        # Extract MAD-RAPID statistics
        module_stats = results.get("module_stats", {})
        mad_rapid_stats = module_stats.get("mad_rapid", {})
        
        # Process MAD-RAPID stats
        mad_rapid_total_tx += mad_rapid_stats.get("total_tx_processed", 0)
        mad_rapid_optimized_tx += mad_rapid_stats.get("optimized_tx_count", 0)
        mad_rapid_latency_improvements += mad_rapid_stats.get("latency_improvement", 0)
        mad_rapid_energy_saved += mad_rapid_stats.get("energy_saved", 0)
        
        # Extract ACSC statistics
        acsc_stats = module_stats.get("acsc", {})
        acsc_total_tx += acsc_stats.get("total_tx_processed", 0)
        
        # Extract strategy usage if available
        strategy_usage = acsc_stats.get("strategy_usage", {})
        acsc_fast_consensus += strategy_usage.get("fast", 0)
        acsc_standard_consensus += strategy_usage.get("standard", 0)
        acsc_robust_consensus += strategy_usage.get("robust", 0)
    
    # Calculate average success rates
    if success_rates:
        aggregate_results["transaction_stats"]["success_rate"] = np.mean(success_rates)
        aggregate_results["transaction_stats"]["success_rate_std"] = np.std(success_rates)
        
        # Calculate 95% confidence interval if we have enough samples
        if len(success_rates) > 1:
            ci = stats.t.interval(0.95, len(success_rates)-1, loc=np.mean(success_rates), scale=stats.sem(success_rates))
            aggregate_results["transaction_stats"]["success_rate_95ci"] = list(ci)
    
    # Calculate cross-shard success rate
    if cross_shard_success_rates:
        aggregate_results["transaction_stats"]["cross_shard_success_rate"] = np.mean(cross_shard_success_rates)
        aggregate_results["transaction_stats"]["cross_shard_success_rate_std"] = np.std(cross_shard_success_rates)
        
        # Calculate 95% confidence interval if we have enough samples
        if len(cross_shard_success_rates) > 1:
            ci = stats.t.interval(0.95, len(cross_shard_success_rates)-1, loc=np.mean(cross_shard_success_rates), scale=stats.sem(cross_shard_success_rates))
            aggregate_results["transaction_stats"]["cross_shard_success_rate_95ci"] = list(ci)
    
    # Calculate average performance metrics
    if throughputs:
        aggregate_results["performance_stats"]["avg_throughput"] = np.mean(throughputs)
    
    if latencies:
        aggregate_results["performance_stats"]["avg_latency"] = np.mean(latencies)
    
    if energy_consumptions:
        aggregate_results["performance_stats"]["energy_consumption"] = np.mean(energy_consumptions)
    
    if congestion_levels:
        aggregate_results["performance_stats"]["congestion_level"] = np.mean(congestion_levels)
    
    # Update MAD-RAPID statistics
    aggregate_results["module_stats"]["mad_rapid"]["total_tx"] = mad_rapid_total_tx
    aggregate_results["module_stats"]["mad_rapid"]["optimized_tx"] = mad_rapid_optimized_tx
    aggregate_results["module_stats"]["mad_rapid"]["latency_improvement"] = mad_rapid_latency_improvements
    aggregate_results["module_stats"]["mad_rapid"]["energy_saved"] = mad_rapid_energy_saved
    
    # Update ACSC statistics
    aggregate_results["module_stats"]["acsc"]["total_tx"] = acsc_total_tx
    aggregate_results["module_stats"]["acsc"]["fast_consensus"] = acsc_fast_consensus
    aggregate_results["module_stats"]["acsc"]["standard_consensus"] = acsc_standard_consensus
    aggregate_results["module_stats"]["acsc"]["robust_consensus"] = acsc_robust_consensus
    
    # Calculate consensus success rate if there were any transactions
    if acsc_total_tx > 0:
        consensus_success = (acsc_fast_consensus + acsc_standard_consensus + acsc_robust_consensus) / acsc_total_tx
        aggregate_results["module_stats"]["acsc"]["consensus_success_rate"] = consensus_success
    
    # In ra thông tin debug
    print("\nEnergy saved in MAD-RAPID:", mad_rapid_energy_saved)
    
    return aggregate_results


if __name__ == "__main__":
    sys.exit(main()) 