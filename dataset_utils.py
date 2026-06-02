from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

NORMAL_NAMES = {
    "normal",
    "benign",
    "benigntraffic",
    "benign_traffic",
    "benign traffic",
    "0",
}

LEAKAGE_COLS = {
    "label", "Label",
    "attack_cat", "Attack_cat",
    "attack_type", "Attack_type",
    "category", "Category",
    "class", "Class",
    "binary_label", "target", "Target",
}


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def normalize_label(series: pd.Series) -> pd.Series:
    """Convert labels to 0=Normal, 1=Attack."""
    s = series.astype(str).str.strip().str.lower()
    return pd.Series(np.where(s.isin(NORMAL_NAMES), 0, 1), index=series.index, name="binary_label")


def label_name(binary: int) -> str:
    return "Attack" if int(binary) == 1 else "Normal"


def load_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"找不到資料檔：{path}")
    return pd.read_csv(path, nrows=nrows)


def find_target_col(df: pd.DataFrame, preferred: str = "label") -> str:
    if preferred in df.columns:
        return preferred
    for c in ["label", "Label", "class", "Class", "target", "Target", "attack_cat", "Attack_type"]:
        if c in df.columns:
            return c
    raise ValueError(f"找不到標籤欄位。目前欄位：{list(df.columns)}")


def clean_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_cols = set(LEAKAGE_COLS)
    drop_cols.add(target_col)
    existing = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=existing, errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


def align_features(X: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = list(feature_names)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]


def prepare_xy(df: pd.DataFrame, target_col: str, feature_names: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    y = normalize_label(df[target_col])
    X = clean_features(df, target_col)
    if feature_names is not None:
        X = align_features(X, feature_names)
    return X, y


def balance_binary_dataset(df: pd.DataFrame, target_col: str, normal_to_attack_ratio: float = 1.0,
                           max_rows_per_class: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    y = normalize_label(df[target_col])
    normal_df = df[y == 0]
    attack_df = df[y == 1]

    if len(normal_df) == 0 or len(attack_df) == 0:
        raise ValueError("資料必須同時包含 Normal 與 Attack，才能做平衡訓練。")

    n_normal = len(normal_df)
    n_attack = int(n_normal / normal_to_attack_ratio) if normal_to_attack_ratio > 0 else len(attack_df)
    n_attack = min(n_attack, len(attack_df))

    if max_rows_per_class:
        n_normal = min(n_normal, max_rows_per_class)
        n_attack = min(n_attack, max_rows_per_class)

    sampled_normal = normal_df.sample(n=n_normal, random_state=random_state, replace=False)
    sampled_attack = attack_df.sample(n=n_attack, random_state=random_state, replace=False)
    out = pd.concat([sampled_normal, sampled_attack], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out


def evaluate_predictions(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "normal_precision": float(precision[0]),
        "normal_recall": float(recall[0]),
        "normal_f1": float(f1[0]),
        "attack_precision": float(precision[1]),
        "attack_recall": float(recall[1]),
        "attack_f1": float(f1[1]),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "normal_support": int(support[0]),
        "attack_support": int(support[1]),
    }


def scan_thresholds(y_true: pd.Series, y_prob: np.ndarray, thresholds: Iterable[float],
                    min_attack_recall: float = 0.0, metric: str = "macro_f1") -> pd.DataFrame:
    rows = [evaluate_predictions(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(rows)
    ok = df[df["attack_recall"] >= min_attack_recall].copy()
    if ok.empty:
        ok = df.copy()
    ok = ok.sort_values([metric, "normal_recall", "attack_recall"], ascending=False)
    df["selected_candidate"] = False
    if not ok.empty:
        df.loc[ok.index[0], "selected_candidate"] = True
    return df.sort_values("threshold")


def selected_threshold_from_scan(scan_df: pd.DataFrame) -> float:
    row = scan_df[scan_df["selected_candidate"] == True]
    if row.empty:
        row = scan_df.sort_values("macro_f1", ascending=False).head(1)
    return float(row.iloc[0]["threshold"])


def print_evaluation(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    lines = []
    lines.append(f"Threshold: {threshold:.2f}")
    lines.append(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} ({accuracy_score(y_true, y_pred)*100:.2f}%)")
    lines.append("\nClassification report:")
    lines.append(classification_report(y_true, y_pred, labels=[0, 1], target_names=["Normal", "Attack"], digits=4, zero_division=0))
    lines.append("Confusion matrix:")
    lines.append(str(confusion_matrix(y_true, y_pred, labels=[0, 1])))
    if len(np.unique(y_true)) == 2:
        try:
            lines.append(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
        except Exception as exc:
            lines.append(f"ROC-AUC: 無法計算 ({exc})")
    else:
        lines.append("ROC-AUC: 無法計算，因為只有一種類別。")
    lines.append("\nConfusion matrix 格式：")
    lines.append("[[正常判斷正常, 正常誤判攻擊],")
    lines.append(" [攻擊誤判正常, 攻擊判斷攻擊]]")
    return "\n".join(lines)


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_distribution(series: pd.Series) -> pd.Series:
    return series.value_counts()
