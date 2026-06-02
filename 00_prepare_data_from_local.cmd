@echo off
chcp 65001 >nul
REM 這支會從你目前的 CICIoT2023 資料夾複製三個 CSV，並改成標準名稱。
REM 如果你的資料夾位置不同，請修改下面 SOURCE_DIR。
set SOURCE_DIR=C:\Users\zi xuan\unb-cic-iot-dataset\wataiData\csv\CICIoT2023
python build_three_split_from_parts.py --source-dir "%SOURCE_DIR%"
pause
