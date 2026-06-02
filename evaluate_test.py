from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from config import MODEL_PATH, REPORT_DIR, TARGET_COL, TEST_CSV, TEST_REPORT_PATH
from dataset_utils import ensure_dirs, find_target_col, load_csv, prepare_xy, print_evaluation, save_json, load_json
from config import META_PATH


def main():
    parser = argparse.ArgumentParser(description="嚴謹版：只用 final test CSV 做最後一次評估，不選 threshold。")
    parser.add_argument("--data", default=str(TEST_CSV), help="最終測試 CSV，例如 data/ciciot2023_test.csv")
    parser.add_argument("--target", default=TARGET_COL, help="標籤欄位名稱")
    parser.add_argument("--model", default=str(MODEL_PATH), help="模型路徑")
    parser.add_argument("--threshold", type=float, default=None, help="可手動指定 threshold；正式實驗建議使用 validation 存好的 threshold")
    args = parser.parse_args()

    ensure_dirs(REPORT_DIR)
    print("讀取最終測試資料中...")
    df = load_csv(Path(args.data))
    target_col = find_target_col(df, args.target)
    print(f"使用標籤欄位: {target_col}")
    print(f"最終測試資料筆數: {len(df)}")
    print("最終測試資料原始標籤分布:")
    print(df[target_col].value_counts())

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    threshold = args.threshold if args.threshold is not None else bundle.get("threshold")
    if threshold is None:
        raise ValueError("模型尚未有 threshold。請先執行 select_threshold.py / 02_select_threshold.cmd。")

    X_test, y_test = prepare_xy(df, target_col, feature_names=feature_names)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n使用 validation 選出的固定 threshold = {threshold:.2f}")
    print("注意：final test 只評估，不再掃描或調整 threshold。")
    result_text = print_evaluation(y_test, prob, threshold)

    print("\n=== 最終測試結果 ===")
    print(result_text)

    TEST_REPORT_PATH.write_text(result_text, encoding="utf-8")

    meta = load_json(META_PATH) if META_PATH.exists() else {}
    meta.update({
        "final_test_data_path": str(Path(args.data)),
        "final_test_threshold": float(threshold),
        "final_test_report_path": str(TEST_REPORT_PATH),
    })
    save_json(META_PATH, meta)
    print(f"\n最終測試報告已儲存: {TEST_REPORT_PATH}")


if __name__ == "__main__":
    main()
