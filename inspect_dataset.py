from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from config import TARGET_COL
from dataset_utils import find_target_col, normalize_label


def main():
    parser = argparse.ArgumentParser(description="檢查 CSV 欄位與標籤分布")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default=TARGET_COL)
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    target_col = find_target_col(df, args.target)
    print(f"資料路徑: {Path(args.data)}")
    print(f"資料筆數: {len(df)}")
    print(f"欄位數: {len(df.columns)}")
    print("\n欄位列表:")
    print(list(df.columns))
    print(f"\n使用標籤欄位: {target_col}")
    print("原始標籤分布:")
    print(df[target_col].value_counts())
    print("\n二元標籤分布:")
    print(normalize_label(df[target_col]).map({0: "Normal", 1: "Attack"}).value_counts())
    print("\n前幾筆資料:")
    print(df.head(args.head))


if __name__ == "__main__":
    main()
