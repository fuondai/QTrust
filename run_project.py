#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DQN Blockchain Simulation Project Runner

This script provides an interactive menu to run various aspects of the
DQN Blockchain Simulation project using the correct Python path.
"""

import os
import sys
import subprocess
import platform
import argparse

def get_python_path():
    """
    Get the full path to the Python executable
    """
    return sys.executable

def install_requirements():
    """
    Install required libraries
    """
    python_path = get_python_path()
    print("=== Installing Required Libraries ===")
    subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    print()

def test_imports():
    """
    Run the import test script
    """
    python_path = get_python_path()
    print("=== Testing Imports ===")
    subprocess.run([python_path, "test_imports.py"], check=True)
    print()

def run_command(command_args):
    """
    Run a command with the full Python path
    
    Args:
        command_args (list): List of command arguments
    """
    python_path = get_python_path()
    full_command = [python_path] + command_args
    print(f"Running: {' '.join(full_command)}")
    subprocess.run(full_command, check=True)
    print()

def show_menu():
    """
    Display the menu of available commands
    """
    print("=== Available Commands ===")
    print("1. Run Basic Simulation")
    print("2. Run Advanced Simulation (4 shards, 10 steps)")
    print("3. Run Advanced Simulation with Visualization")
    print("4. Run Advanced Simulation with Stats Saving")
    print("5. Run Consensus Comparison")
    print("6. Run Performance Analysis")
    print("7. Generate Report")
    print("8. Run CLI Analysis")
    print("9. Run GUI Analysis")
    print("0. Exit")
    print()

def main():
    """
    Main function to run the interactive menu
    """
    parser = argparse.ArgumentParser(description="DQN Blockchain Simulation Project Runner")
    parser.add_argument('--skip-install', action='store_true', help="Skip installing requirements")
    parser.add_argument('--skip-tests', action='store_true', help="Skip import tests")
    args = parser.parse_args()
    
    # Print Python information
    python_path = get_python_path()
    print(f"Using Python: {python_path}")
    print(f"Python version: {platform.python_version()}")
    print()
    
    # Install requirements if not skipped
    if not args.skip_install:
        try:
            install_requirements()
        except subprocess.CalledProcessError:
            print("Warning: Failed to install requirements")
    
    # Test imports if not skipped
    if not args.skip_tests:
        try:
            test_imports()
        except subprocess.CalledProcessError:
            print("Warning: Import tests failed")
    
    # Command mapping
    commands = {
        "1": ["-m", "dqn_blockchain_sim.simulation.main"],
        "2": ["-m", "dqn_blockchain_sim.run_advanced_simulation", "--num_shards", "4", "--steps", "10", "--tx_per_step", "5"],
        "3": ["-m", "dqn_blockchain_sim.run_advanced_simulation", "--num_shards", "4", "--steps", "10", "--tx_per_step", "5", "--visualize"],
        "4": ["-m", "dqn_blockchain_sim.run_advanced_simulation", "--num_shards", "4", "--steps", "10", "--tx_per_step", "5", "--save_stats"],
        "5": ["-m", "dqn_blockchain_sim.experiments.consensus_comparison"],
        "6": ["-m", "dqn_blockchain_sim.experiments.performance_analysis"],
        "7": ["-m", "dqn_blockchain_sim.experiments.generate_report"],
        "8": ["-m", "dqn_blockchain_sim.run_analysis"],
        "9": ["-m", "dqn_blockchain_sim.run_analysis_gui"]
    }
    
    # Menu labels
    labels = {
        "1": "Running Basic Simulation",
        "2": "Running Advanced Simulation",
        "3": "Running Advanced Simulation with Visualization",
        "4": "Running Advanced Simulation with Stats Saving",
        "5": "Running Consensus Comparison",
        "6": "Running Performance Analysis",
        "7": "Generating Report",
        "8": "Running CLI Analysis",
        "9": "Running GUI Analysis"
    }
    
    # Interactive menu
    while True:
        show_menu()
        choice = input("Enter your choice (0-9): ")
        
        if choice == "0":
            print("Exiting program")
            break
        
        if choice in commands:
            print(f"=== {labels[choice]} ===")
            try:
                run_command(commands[choice])
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
        else:
            print("Invalid choice!")
        
        print()

if __name__ == "__main__":
    main() 