@echo off
REM Script to run DQN Blockchain Simulation with full Python path

REM Get Python path from environment PATH or use default
where python > nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    set PYTHON_PATH=c:\users\dadad\appdata\local\programs\python\python310\python.exe
) ELSE (
    for /f "delims=" %%i in ('where python') do set PYTHON_PATH=%%i
)

echo Using Python: %PYTHON_PATH%
echo.

REM Install required libraries
echo === Installing Required Libraries ===
%PYTHON_PATH% -m pip install -r requirements.txt
echo.

REM Run import tests
echo === Testing Imports ===
%PYTHON_PATH% test_imports.py
echo.

REM List of available commands
echo === Available Commands ===
echo 1. Run Basic Simulation
echo 2. Run Advanced Simulation (4 shards, 10 steps)
echo 3. Run Advanced Simulation with Visualization
echo 4. Run Advanced Simulation with Stats Saving
echo 5. Run Consensus Comparison
echo 6. Run Performance Analysis
echo 7. Generate Report
echo 8. Run CLI Analysis
echo 9. Run GUI Analysis
echo 0. Exit
echo.

:menu
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" (
    echo === Running Basic Simulation ===
    %PYTHON_PATH% -m dqn_blockchain_sim.simulation.main
) else if "%choice%"=="2" (
    echo === Running Advanced Simulation ===
    %PYTHON_PATH% -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5
) else if "%choice%"=="3" (
    echo === Running Advanced Simulation with Visualization ===
    %PYTHON_PATH% -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --visualize
) else if "%choice%"=="4" (
    echo === Running Advanced Simulation with Stats Saving ===
    %PYTHON_PATH% -m dqn_blockchain_sim.run_advanced_simulation --num_shards 4 --steps 10 --tx_per_step 5 --save_stats
) else if "%choice%"=="5" (
    echo === Running Consensus Comparison ===
    %PYTHON_PATH% -m dqn_blockchain_sim.experiments.consensus_comparison
) else if "%choice%"=="6" (
    echo === Running Performance Analysis ===
    %PYTHON_PATH% -m dqn_blockchain_sim.experiments.performance_analysis
) else if "%choice%"=="7" (
    echo === Generating Report ===
    %PYTHON_PATH% -m dqn_blockchain_sim.experiments.generate_report
) else if "%choice%"=="8" (
    echo === Running CLI Analysis ===
    %PYTHON_PATH% -m dqn_blockchain_sim.run_analysis
) else if "%choice%"=="9" (
    echo === Running GUI Analysis ===
    %PYTHON_PATH% -m dqn_blockchain_sim.run_analysis_gui
) else if "%choice%"=="0" (
    echo Exiting program
    goto :eof
) else (
    echo Invalid choice!
)

echo.
goto menu 