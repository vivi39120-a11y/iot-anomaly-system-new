@echo off
chcp 65001 >nul
python select_threshold.py --data data\ciciot2023_val.csv --target label
pause
