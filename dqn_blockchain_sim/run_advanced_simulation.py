#!/usr/bin/env python
"""
Script to run advanced blockchain simulation with improvements
"""

import os
import sys
import argparse
from dqn_blockchain_sim.simulation.advanced_simulation import AdvancedSimulation


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
    parser.add_argument('--steps', type=int, default=500, 
                        help='Number of simulation steps (default: 500)')
    parser.add_argument('--tx_per_step', type=int, default=20, 
                        help='Number of transactions per step (default: 20)')
    
    # Module options
    parser.add_argument('--use_real_data', action='store_true', 
                        help='Use real Ethereum data')
    parser.add_argument('--use_dqn', action='store_true', default=True,
                        help='Use DQN agent (default: True)')
    parser.add_argument('--eth_api_key', type=str, 
                        help='Ethereum API key (required if --use_real_data)')
    
    # Output options
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Data directory (default: "data")')
    parser.add_argument('--log_dir', type=str, default='logs', 
                        help='Log directory (default: "logs")')
    parser.add_argument('--visualize', action='store_true', 
                        help='Display result charts')
    parser.add_argument('--save_stats', action='store_true', 
                        help='Save statistics to file')
    
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
    print(f"Using real data: {args.use_real_data}")
    print(f"Using DQN agent: {args.use_dqn}")
    
    # Initialize simulation
    simulation = AdvancedSimulation(
        num_shards=args.num_shards,
        use_real_data=args.use_real_data,
        use_dqn=args.use_dqn,
        eth_api_key=args.eth_api_key,
        data_dir=args.data_dir,
        log_dir=args.log_dir
    )
    
    # Run simulation
    results = simulation.run_simulation(
        num_steps=args.steps,
        tx_per_step=args.tx_per_step,
        visualize=args.visualize,
        save_stats=args.save_stats
    )
    
    # Display summary results
    print("\n===== SIMULATION RESULTS =====")
    if 'transaction_stats' in results and isinstance(results['transaction_stats'], dict):
        tx_stats = results['transaction_stats']
        print(f"Total transactions: {tx_stats.get('total_transactions', 'N/A')}")
        print(f"Successful transactions: {tx_stats.get('successful_transactions', 'N/A')}")
        print(f"Success rate: {tx_stats.get('success_rate', 0):.2%}")
    else:
        print("Không có thống kê giao dịch chi tiết")
    
    if 'performance_metrics' in results:
        metrics = results['performance_metrics']
        print("\n--- AVERAGE PERFORMANCE ---")
        print(f"Throughput: {metrics.get('avg_throughput', 0):.2f} transactions/step")
        print(f"Latency: {metrics.get('avg_latency', 0):.2f} ms")
        print(f"Congestion level: {metrics.get('avg_congestion', 0):.2%}")
        print(f"Energy consumption: {metrics.get('avg_energy_consumption', 0):.2f} units")
    
    print("\n=== MODULE STATISTICS ===")
    
    # MAD-RAPID
    if 'module_stats' in results and 'mad_rapid' in results['module_stats']:
        mad_rapid_stats = results['module_stats']['mad_rapid']
        print("\nMAD-RAPID Protocol:")
        print(f"- Total cross-shard transactions: {mad_rapid_stats.get('total_cross_shard_transactions', 0)}")
        print(f"- Optimized transactions: {mad_rapid_stats.get('optimized_transactions', 0)}")
        latency_imp = mad_rapid_stats.get('latency_improvement_percent', 0)
        print(f"- Latency improvement: {latency_imp:.2f}%")
        print(f"- Energy saved: {mad_rapid_stats.get('energy_saved', 0):.2f} units")
    
    # HTDCM
    if 'module_stats' in results and 'htdcm' in results['module_stats']:
        htdcm_stats = results['module_stats']['htdcm']
        print("\nHTDCM:")
        print(f"- Average trust score: {htdcm_stats.get('avg_trust_score', 0):.2f}")
        print(f"- Trust violations detected: {htdcm_stats.get('trust_violations_detected', 0)}")
        tx_security = htdcm_stats.get('transaction_security_level', {})
        print(f"- High security transactions: {tx_security.get('high', 0)}")
        print(f"- Medium security transactions: {tx_security.get('medium', 0)}")
        print(f"- Low security transactions: {tx_security.get('low', 0)}")
    
    # ACSC
    if 'module_stats' in results and 'acsc' in results['module_stats']:
        acsc_stats = results['module_stats']['acsc']
        print("\nACSC:")
        print(f"- Total transactions: {acsc_stats.get('total_transactions', 0)}")
        print(f"- Fast Consensus usage: {acsc_stats.get('fast_consensus_percent', 0):.2f}%")
        print(f"- Standard Consensus usage: {acsc_stats.get('standard_consensus_percent', 0):.2f}%")
        print(f"- Robust Consensus usage: {acsc_stats.get('robust_consensus_percent', 0):.2f}%")
        print(f"- Consensus success rate: {acsc_stats.get('consensus_success_rate', 0):.2f}%")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 