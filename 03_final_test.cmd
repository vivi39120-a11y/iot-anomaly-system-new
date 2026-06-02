@echo off
chcp 65001 >nul
python evaluate_test.py --data data\ciciot2023_test.csv --target label
pause
