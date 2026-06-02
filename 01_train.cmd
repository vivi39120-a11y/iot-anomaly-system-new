@echo off
chcp 65001 >nul
python -m pip install -r requirements.txt
python train_model.py --data data\ciciot2023_train.csv --target label
pause
