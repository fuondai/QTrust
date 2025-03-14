"""
DQN Blockchain Simulation - Mô phỏng blockchain tối ưu với Deep Q-Network
"""

# Import các module chính
from dqn_blockchain_sim.simulation import advanced_simulation
from dqn_blockchain_sim.blockchain import network, transaction, shard, mad_rapid, adaptive_consensus
from dqn_blockchain_sim.agents import dqn_agent, enhanced_dqn
from dqn_blockchain_sim.tdcm import trust_manager
from dqn_blockchain_sim.federated_learning import federated_learner
from dqn_blockchain_sim.federated import federated_learning

__version__ = '1.0.0'
__author__ = 'DQN Blockchain Team' 