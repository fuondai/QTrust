#!/usr/bin/env python
# coding: utf-8

"""
Script kiểm tra các import trong dqn_blockchain_sim
"""

import sys
import traceback

def test_imports():
    """Kiểm tra các import chính"""
    # Import blockchain.network
    print("Import BlockchainNetwork từ blockchain.network...")
    try:
        from dqn_blockchain_sim.blockchain.network import BlockchainNetwork
        print("  ✓ Thành công")
        
        # Kiểm tra khởi tạo BlockchainNetwork
        print("Khởi tạo BlockchainNetwork...")
        config = {"num_shards": 4, "num_nodes": 40, "min_nodes_per_shard": 3, "block_time": 2.0}
        network = BlockchainNetwork(config=config)
        print("  ✓ Thành công")
    except Exception as e:
        print(f"  ✗ Lỗi: {e}")
        traceback.print_exc()
    
    # Import các module tích hợp
    modules = [
        ("dqn_blockchain_sim.blockchain.mad_rapid", "MAD_RAPID_Protocol"),
        ("dqn_blockchain_sim.tdcm.hierarchical_trust", "HierarchicalTrustDCM"),
        ("dqn_blockchain_sim.blockchain.adaptive_consensus", "AdaptiveCrossShardConsensus"),
        ("dqn_blockchain_sim.utils.real_data_builder", "EthereumDataProcessor"),
        ("dqn_blockchain_sim.simulation.mad_rapid_integration", "MADRAPIDIntegration"),
        ("dqn_blockchain_sim.simulation.htdcm_integration", "HTDCMIntegration"),
        ("dqn_blockchain_sim.simulation.acsc_integration", "ACSCIntegration"),
        ("dqn_blockchain_sim.simulation.real_data_integration", "RealDataIntegration"),
        ("dqn_blockchain_sim.agents.dqn_agent", "ShardDQNAgent")
    ]
    
    for module_path, class_name in modules:
        print(f"Import {class_name} từ {module_path}...")
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"  ✓ Thành công")
        except ImportError as e:
            print(f"  ✗ Lỗi import: {e}")
        except AttributeError as e:
            print(f"  ✗ Lỗi thuộc tính: {e}")
        except Exception as e:
            print(f"  ✗ Lỗi khác: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.path}")
    test_imports() 