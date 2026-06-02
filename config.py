from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TRAIN_CSV = DATA_DIR / "ciciot2023_train.csv"
VAL_CSV = DATA_DIR / "ciciot2023_val.csv"
TEST_CSV = DATA_DIR / "ciciot2023_test.csv"

MODEL_PATH = MODEL_DIR / "xgb_iot_detector.joblib"
META_PATH = MODEL_DIR / "training_meta.json"
VAL_REPORT_PATH = REPORT_DIR / "validation_threshold_report.csv"
TEST_REPORT_PATH = REPORT_DIR / "final_test_report.txt"

TARGET_COL = "label"
RANDOM_STATE = 42

# 平衡訓練設定：避免攻擊資料太多導致模型只會猜 Attack
BALANCE_TRAINING = True
NORMAL_TO_ATTACK_RATIO = 1.0  # 1.0 = Normal:Attack 約 1:1
MAX_TRAIN_ROWS_PER_CLASS = 60000

# Threshold 選擇設定
# validation 階段會掃描門檻，只用 validation 選 threshold；final test 不再調 threshold
THRESHOLD_GRID = [round(x / 100, 2) for x in range(5, 100, 5)]
THRESHOLD_SELECTION_METRIC = "macro_f1"  # macro_f1 較能兼顧 normal/attack
MIN_ATTACK_RECALL = 0.90                # 避免門檻太高導致攻擊漏抓太多

# Streamlit 模擬顯示設定：讓正常流量成為主要背景
SIMULATION_MODES = {
    "日常監控（正常多，少量異常）": 0.05,
    "可疑活動增加": 0.15,
    "攻擊爆發時段": 0.30,
}
