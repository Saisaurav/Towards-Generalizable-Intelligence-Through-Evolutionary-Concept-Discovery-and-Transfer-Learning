@echo off
setlocal
cd /d "%~dp0"
python install_builtin_envs.py
if errorlevel 1 exit /b 1
pushd src
where g++ >nul 2>nul
if errorlevel 1 (
  echo g++ was not found. Install MinGW-w64 or use WSL, then rerun.
  popd
  exit /b 1
)
g++ -std=c++17 -O2 -pthread -Wall -Wextra main.cpp -o brain.exe
if errorlevel 1 (popd & exit /b 1)
popd
python train_long_term_suite.py --games all --trials 3 --minutes 45 --resume
if errorlevel 1 exit /b 1
python deploy_trained_policies.py --games all
