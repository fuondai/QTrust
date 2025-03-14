#!/bin/bash
# Script to run DQN Blockchain Simulation with full Python path

# Get Python path from environment PATH
PYTHON_PATH=$(which python3 2>/dev/null || which python)

if [ -z "$PYTHON_PATH" ]; then
    echo "Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

echo "Using Python: $PYTHON_PATH"
echo

# Install required libraries
echo "=== Installing Required Libraries ==="
$PYTHON_PATH -m pip install -r requirements.txt
echo

# Run import tests
echo "=== Testing Imports ==="
$PYTHON_PATH test_imports.py
echo

# List of available commands
show_menu() {
    echo "=== Available Commands ==="
    echo "1. Run Basic Simulation"
    echo "2. Run Advanced Simulation (4 shards, 10 steps)"
    echo "3. Run Advanced Simulation with Visualization"
    echo "4. Run Advanced Simulation with Stats Saving"
    echo "5. Run Consensus Comparison"
    echo "6. Run Performance Analysis"
    echo "7. Generate Report"
    echo "8. Run CLI Analysis"
    echo "9. Run GUI Analysis"
    echo "0. Exit"
    echo
}

# Loop menu
while true; do
    show_menu
    read -p "Enter your choice (0-9): " choice
    
    case $choice in
        1)
            echo "=== Running Basic Simulation ==="
            $PYTHON_PATH -m dqn_blockchain_sim.simulation.main
            ;;
        2)
            echo "=== Running Advanced Simulation ==="
            $PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5
            ;;
        3)
            echo "=== Running Advanced Simulation with Visualization ==="
            $PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --visualize
            ;;
        4)
            echo "=== Running Advanced Simulation with Stats Saving ==="
            $PYTHON_PATH -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --save_stats
            ;;
        5)
            echo "=== Running Consensus Comparison ==="
            $PYTHON_PATH -m dqn_blockchain_sim.experiments.consensus_comparison
            ;;
        6)
            echo "=== Running Performance Analysis ==="
            $PYTHON_PATH -m dqn_blockchain_sim.experiments.performance_analysis
            ;;
        7)
            echo "=== Generating Report ==="
            $PYTHON_PATH -m dqn_blockchain_sim.experiments.generate_report
            ;;
        8)
            echo "=== Running CLI Analysis ==="
            $PYTHON_PATH -m dqn_blockchain_sim.run_analysis
            ;;
        9)
            echo "=== Running GUI Analysis ==="
            $PYTHON_PATH -m dqn_blockchain_sim.run_analysis_gui
            ;;
        0)
            echo "Exiting program"
            exit 0
            ;;
        *)
            echo "Invalid choice!"
            ;;
    esac
    
    echo
done 