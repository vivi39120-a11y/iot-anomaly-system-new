from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import (
    BALANCE_TRAINING,
    DATA_DIR,
    MAX_TRAIN_ROWS_PER_CLASS,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    NORMAL_TO_ATTACK_RATIO,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_CSV,
)
from dataset_utils import (
    balance_binary_dataset,
    ensure_dirs,
    find_target_col,
    load_csv,
    normalize_label,
    prepare_xy,
    print_evaluation,
    save_json,
    summarize_distribution,
)


def main():
    parser = argparse.ArgumentParser(description="嚴謹版：只用訓練 CSV 訓練模型，不用它選 threshold。")
    parser.add_argument("--data", default=str(TRAIN_CSV), help="訓練 CSV，例如 data/ciciot2023_train.csv")
    parser.add_argument("--target", default=TARGET_COL, help="標籤欄位名稱，CICIoT2023 通常為 label")
    parser.add_argument("--no-balance", action="store_true", help="不要做 normal/attack 平衡抽樣")
    args = parser.parse_args()

    ensure_dirs(DATA_DIR, MODEL_DIR)

    data_path = Path(args.data)
    print("讀取訓練資料中...")
    df = load_csv(data_path)
    target_col = find_target_col(df, args.target)
    print(f"使用標籤欄位: {target_col}")
    print(f"原始訓練資料筆數: {len(df)}")
    print("原始標籤分布:")
    print(summarize_distribution(df[target_col]))

    y_raw = normalize_label(df[target_col])
    print("\n原始二元標籤分布:")
    print(y_raw.map({0: "Normal", 1: "Attack"}).value_counts())

    if BALANCE_TRAINING and not args.no_balance:
        df_train_balanced = balance_binary_dataset(
            df,
            target_col=target_col,
            normal_to_attack_ratio=NORMAL_TO_ATTACK_RATIO,
            max_rows_per_class=MAX_TRAIN_ROWS_PER_CLASS,
            random_state=RANDOM_STATE,
        )
        print("\n平衡後訓練資料筆數:", len(df_train_balanced))
        print("平衡後二元標籤分布:")
        print(normalize_label(df_train_balanced[target_col]).map({0: "Normal", 1: "Attack"}).value_counts())
    else:
        df_train_balanced = df
        print("\n未啟用平衡抽樣。")

    X, y = prepare_xy(df_train_balanced, target_col)
    feature_names = list(X.columns)
    print(f"\n特徵欄位數: {len(feature_names)}")

    X_train, X_internal, y_train, y_internal = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("\n訓練 XGBoost 中...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 注意：這裡只做 sanity check，不用來決定最後 threshold。
    internal_prob = model.predict_proba(X_internal)[:, 1]
    internal_threshold = 0.5
    print("\n=== 內部驗證 sanity check（不作為最終測試結果） ===")
    print(print_evaluation(y_internal, internal_prob, internal_threshold))

    print("\n訓練 Isolation Forest 正常基準線模型中...")
    normal_X = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(normal_X)

    bundle = {
        "model": model,
        "isolation_forest": iso,
        "feature_names": feature_names,
        "target_col": target_col,
        "threshold": None,  # select_threshold.py 才會填入
        "label_mapping": {"Normal": 0, "Attack": 1},
    }
    joblib.dump(bundle, MODEL_PATH)

    meta = {
        "data_path": str(data_path),
        "target_col": target_col,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "balanced_training": bool(BALANCE_TRAINING and not args.no_balance),
        "threshold": None,
        "threshold_source": "not_selected_yet",
        "internal_sanity_check_threshold": internal_threshold,
        "internal_sanity_check_auc": float(roc_auc_score(y_internal, internal_prob)) if len(np.unique(y_internal)) == 2 else None,
        "note": "正式 threshold 請用 validation CSV 執行 select_threshold.py 後產生；final test 請用 evaluate_test.py。",
    }
    save_json(META_PATH, meta)

    print(f"\n模型已儲存: {MODEL_PATH}")
    print(f"訓練資訊已儲存: {META_PATH}")
    print("\n下一步請執行：02_select_threshold.cmd")


if __name__ == "__main__":
    main()
