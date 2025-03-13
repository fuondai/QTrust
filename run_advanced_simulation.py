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
            log_dir=args.log_dir,
            network_config={
                "base_latency": args.network_latency,
                "latency_variation": args.latency_variation,
                "packet_loss_prob": args.packet_loss,
                "block_time": args.block_time
            }
        )

        # Run simulation
        results = simulation.run_simulation(
            num_steps=args.steps,
            tx_per_step=args.tx_per_step,
            visualize=False,  # Don't visualize individual iterations
            save_stats=False  # Don't save individual iterations
        )
        
        # Store iteration results
        results["iteration"] = iteration
        all_results.append(results)
        
        # Show condensed iteration results
        print(f"\n--- Iteration {iteration+1} Results ---")
        print(f"Success rate: {results['transaction_stats']['success_rate']:.2%}")
        print(f"Cross-shard success rate: {results['transaction_stats'].get('cross_shard_success_rate', 0):.2%}")
        if 'performance_metrics' in results:
            print(f"Congestion level: {results['performance_metrics'].get('avg_congestion', 0):.2%}")

    # Calculate aggregate statistics
    aggregate_results = calculate_aggregate_statistics(all_results)
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
        print(f"Congestion level: {metrics.get('avg_congestion', 0):.2%}")
        print(f"Energy consumption: {metrics.get('avg_energy_consumption', 0):.2f} units")

    print("\n=== MODULE STATISTICS ===")

    # MAD-RAPID
    if 'module_stats' in aggregate_results and 'mad_rapid' in aggregate_results['module_stats']:
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
    Calculate aggregate statistics across multiple simulation iterations
    
    Args:
        results_list: List of results from multiple iterations
        
    Returns:
        Dictionary with aggregate statistics
    """
    import numpy as np
    from scipy import stats
    
    # Initialize aggregate results
    aggregate = {
        "transaction_stats": {},
        "performance_metrics": {},
        "module_stats": {
            "mad_rapid": {},
            "htdcm": {},
            "acsc": {}
        },
        "iterations": len(results_list)
    }
    
    # Transaction statistics
    success_rates = []
    cross_shard_success_rates = []
    total_tx = 0
    successful_tx = 0
    cross_shard_tx = 0
    successful_cross_shard_tx = 0
    
    for result in results_list:
        tx_stats = result.get("transaction_stats", {})
        success_rates.append(tx_stats.get("success_rate", 0))
        
        if "cross_shard_success_rate" in tx_stats:
            cross_shard_success_rates.append(tx_stats["cross_shard_success_rate"])
        
        total_tx += tx_stats.get("total_transactions", 0)
        successful_tx += tx_stats.get("successful_transactions", 0)
        cross_shard_tx += tx_stats.get("cross_shard_transactions", 0)
        successful_cross_shard_tx += tx_stats.get("successful_cross_shard_transactions", 0)
    
    # Calculate mean and confidence intervals
    success_rate_mean = np.mean(success_rates)
    success_rate_std = np.std(success_rates)
    success_rate_ci = stats.t.interval(0.95, len(success_rates)-1, 
                                      loc=success_rate_mean, 
                                      scale=success_rate_std/np.sqrt(len(success_rates)))
    
    aggregate["transaction_stats"]["success_rate"] = success_rate_mean
    aggregate["transaction_stats"]["success_rate_std"] = float(success_rate_std)
    aggregate["transaction_stats"]["success_rate_95ci"] = [float(ci) for ci in success_rate_ci]
    aggregate["transaction_stats"]["total_transactions"] = total_tx
    aggregate["transaction_stats"]["successful_transactions"] = successful_tx
    
    if cross_shard_tx > 0:
        cross_shard_rate = successful_cross_shard_tx / cross_shard_tx
        aggregate["transaction_stats"]["cross_shard_transactions"] = cross_shard_tx
        aggregate["transaction_stats"]["successful_cross_shard_transactions"] = successful_cross_shard_tx
        aggregate["transaction_stats"]["cross_shard_success_rate"] = cross_shard_rate
        
        if cross_shard_success_rates:
            cross_shard_mean = np.mean(cross_shard_success_rates)
            cross_shard_std = np.std(cross_shard_success_rates)
            cross_shard_ci = stats.t.interval(0.95, len(cross_shard_success_rates)-1,
                                             loc=cross_shard_mean,
                                             scale=cross_shard_std/np.sqrt(len(cross_shard_success_rates)))
            aggregate["transaction_stats"]["cross_shard_success_rate_std"] = float(cross_shard_std)
            aggregate["transaction_stats"]["cross_shard_success_rate_95ci"] = [float(ci) for ci in cross_shard_ci]
    
    # Performance metrics
    throughputs = []
    latencies = []
    congestions = []
    energy_consumptions = []
    
    for result in results_list:
        if "performance_metrics" in result:
            metrics = result["performance_metrics"]
            throughputs.append(metrics.get("avg_throughput", 0))
            latencies.append(metrics.get("avg_latency", 0))
            congestions.append(metrics.get("avg_congestion", 0))
            energy_consumptions.append(metrics.get("avg_energy_consumption", 0))
    
    if throughputs:
        aggregate["performance_metrics"]["avg_throughput"] = np.mean(throughputs)
        aggregate["performance_metrics"]["throughput_std"] = float(np.std(throughputs))
    
    if latencies:
        aggregate["performance_metrics"]["avg_latency"] = np.mean(latencies)
        aggregate["performance_metrics"]["latency_std"] = float(np.std(latencies))
    
    if congestions:
        aggregate["performance_metrics"]["avg_congestion"] = np.mean(congestions)
        aggregate["performance_metrics"]["congestion_std"] = float(np.std(congestions))
    
    if energy_consumptions:
        aggregate["performance_metrics"]["avg_energy_consumption"] = np.mean(energy_consumptions)
        aggregate["performance_metrics"]["energy_consumption_std"] = float(np.std(energy_consumptions))
    
    # Module statistics
    # MAD-RAPID
    mad_rapid_total_tx = 0
    mad_rapid_optimized_tx = 0
    mad_rapid_latency_improvements = []
    mad_rapid_energy_saved = 0
    
    # HTDCM
    htdcm_trust_scores = []
    htdcm_violations = 0
    security_levels = {"high": 0, "medium": 0, "low": 0}
    
    # ACSC
    acsc_total_tx = 0
    acsc_fast_consensus = []
    acsc_standard_consensus = []
    acsc_robust_consensus = []
    acsc_success_rates = []
    
    for result in results_list:
        if "module_stats" not in result:
            continue
            
        # MAD-RAPID stats
        if "mad_rapid" in result["module_stats"]:
            mr_stats = result["module_stats"]["mad_rapid"]
            mad_rapid_total_tx += mr_stats.get("total_cross_shard_transactions", 0)
            mad_rapid_optimized_tx += mr_stats.get("optimized_transactions", 0)
            if "latency_improvement_percent" in mr_stats:
                mad_rapid_latency_improvements.append(mr_stats["latency_improvement_percent"])
            mad_rapid_energy_saved += mr_stats.get("energy_saved", 0)
        
        # HTDCM stats
        if "htdcm" in result["module_stats"]:
            ht_stats = result["module_stats"]["htdcm"]
            if "avg_trust_score" in ht_stats:
                htdcm_trust_scores.append(ht_stats["avg_trust_score"])
            htdcm_violations += ht_stats.get("trust_violations_detected", 0)
            
            if "transaction_security_level" in ht_stats:
                sec_levels = ht_stats["transaction_security_level"]
                security_levels["high"] += sec_levels.get("high", 0)
                security_levels["medium"] += sec_levels.get("medium", 0)
                security_levels["low"] += sec_levels.get("low", 0)
        
        # ACSC stats
        if "acsc" in result["module_stats"]:
            acsc_stats = result["module_stats"]["acsc"]
            acsc_total_tx += acsc_stats.get("total_transactions", 0)
            
            if "fast_consensus_percent" in acsc_stats:
                acsc_fast_consensus.append(acsc_stats["fast_consensus_percent"])
            if "standard_consensus_percent" in acsc_stats:
                acsc_standard_consensus.append(acsc_stats["standard_consensus_percent"])
            if "robust_consensus_percent" in acsc_stats:
                acsc_robust_consensus.append(acsc_stats["robust_consensus_percent"])
            if "consensus_success_rate" in acsc_stats:
                acsc_success_rates.append(acsc_stats["consensus_success_rate"])
    
    # MAD-RAPID aggregate stats
    if mad_rapid_total_tx > 0:
        aggregate["module_stats"]["mad_rapid"]["total_cross_shard_transactions"] = mad_rapid_total_tx
        aggregate["module_stats"]["mad_rapid"]["optimized_transactions"] = mad_rapid_optimized_tx
        if mad_rapid_latency_improvements:
            aggregate["module_stats"]["mad_rapid"]["latency_improvement_percent"] = np.mean(mad_rapid_latency_improvements)
        aggregate["module_stats"]["mad_rapid"]["energy_saved"] = mad_rapid_energy_saved
    
    # HTDCM aggregate stats
    if htdcm_trust_scores:
        aggregate["module_stats"]["htdcm"]["avg_trust_score"] = np.mean(htdcm_trust_scores)
    aggregate["module_stats"]["htdcm"]["trust_violations_detected"] = htdcm_violations
    aggregate["module_stats"]["htdcm"]["transaction_security_level"] = security_levels
    
    # ACSC aggregate stats
    if acsc_total_tx > 0:
        aggregate["module_stats"]["acsc"]["total_transactions"] = acsc_total_tx
        if acsc_fast_consensus:
            aggregate["module_stats"]["acsc"]["fast_consensus_percent"] = np.mean(acsc_fast_consensus)
        if acsc_standard_consensus:
            aggregate["module_stats"]["acsc"]["standard_consensus_percent"] = np.mean(acsc_standard_consensus)
        if acsc_robust_consensus:
            aggregate["module_stats"]["acsc"]["robust_consensus_percent"] = np.mean(acsc_robust_consensus)
        if acsc_success_rates:
            aggregate["module_stats"]["acsc"]["consensus_success_rate"] = np.mean(acsc_success_rates)
    
    return aggregate


if __name__ == "__main__":
    sys.exit(main()) 