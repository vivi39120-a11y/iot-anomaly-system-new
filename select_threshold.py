from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from config import (
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    REPORT_DIR,
    TARGET_COL,
    THRESHOLD_GRID,
    THRESHOLD_SELECTION_METRIC,
    MIN_ATTACK_RECALL,
    VAL_CSV,
    VAL_REPORT_PATH,
)
from dataset_utils import (
    ensure_dirs,
    find_target_col,
    load_csv,
    load_json,
    prepare_xy,
    print_evaluation,
    save_json,
    scan_thresholds,
    selected_threshold_from_scan,
)


def main():
    parser = argparse.ArgumentParser(description="嚴謹版：只用 validation CSV 選 threshold。")
    parser.add_argument("--data", default=str(VAL_CSV), help="驗證 CSV，例如 data/ciciot2023_val.csv")
    parser.add_argument("--target", default=TARGET_COL, help="標籤欄位名稱")
    parser.add_argument("--model", default=str(MODEL_PATH), help="模型路徑")
    args = parser.parse_args()

    ensure_dirs(REPORT_DIR, MODEL_DIR)
    print("讀取驗證資料中...")
    df = load_csv(Path(args.data))
    target_col = find_target_col(df, args.target)
    print(f"使用標籤欄位: {target_col}")
    print(f"驗證資料筆數: {len(df)}")
    print("驗證資料原始標籤分布:")
    print(df[target_col].value_counts())

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    X_val, y_val = prepare_xy(df, target_col, feature_names=feature_names)

    print("\n用 validation 資料掃描 threshold...")
    prob = model.predict_proba(X_val)[:, 1]
    scan_df = scan_thresholds(
        y_val,
        prob,
        thresholds=THRESHOLD_GRID,
        min_attack_recall=MIN_ATTACK_RECALL,
        metric=THRESHOLD_SELECTION_METRIC,
    )
    selected_threshold = selected_threshold_from_scan(scan_df)
    scan_df.to_csv(VAL_REPORT_PATH, index=False, encoding="utf-8-sig")

    print("\n=== Validation threshold 掃描結果 Top 10 ===")
    cols = ["threshold", "accuracy", "normal_recall", "normal_f1", "attack_recall", "attack_f1", "macro_f1", "selected_candidate"]
    print(scan_df.sort_values("macro_f1", ascending=False)[cols].head(10).to_string(index=False))

    print(f"\n選定 threshold = {selected_threshold:.2f}")
    print("選擇原則：只用 validation 資料選 threshold；final test 不再調整 threshold。")
    print("\n=== Validation 使用選定 threshold 的結果 ===")
    print(print_evaluation(y_val, prob, selected_threshold))

    bundle["threshold"] = selected_threshold
    joblib.dump(bundle, args.model)

    meta = load_json(META_PATH) if META_PATH.exists() else {}
    meta.update({
        "threshold": selected_threshold,
        "threshold_source": "validation_csv",
        "validation_data_path": str(Path(args.data)),
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "min_attack_recall": MIN_ATTACK_RECALL,
        "validation_report_path": str(VAL_REPORT_PATH),
    })
    save_json(META_PATH, meta)

    print(f"\nthreshold 已寫入模型與 meta: {META_PATH}")
    print("下一步請執行：03_final_test.cmd")


if __name__ == "__main__":
    main()
