from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="把三個 CICIoT2023 part CSV 複製成 train/val/test 標準檔名")
    parser.add_argument("--source-dir", required=True, help="CICIoT2023 CSV 資料夾")
    parser.add_argument("--train", default="part-00092-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
    parser.add_argument("--val", default="part-00037-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
    parser.add_argument("--test", default="part-00014-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
    args = parser.parse_args()

    src = Path(args.source_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    pairs = [
        (src / args.train, DATA_DIR / "ciciot2023_train.csv"),
        (src / args.val, DATA_DIR / "ciciot2023_val.csv"),
        (src / args.test, DATA_DIR / "ciciot2023_test.csv"),
    ]

    for s, d in pairs:
        if not s.exists():
            raise FileNotFoundError(f"找不到來源檔案：{s}")
        print(f"複製 {s.name} -> {d}")
        shutil.copy2(s, d)

    print("\n完成。接著依序執行 01_train.cmd、02_select_threshold.cmd、03_final_test.cmd、04_run_app.cmd")


if __name__ == "__main__":
    main()
