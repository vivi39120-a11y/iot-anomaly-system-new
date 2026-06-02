# IoT 異常偵測模擬系統：嚴謹三資料集版

本版本採用最嚴謹的三段流程，避免「用測試資料調參數後再回報測試準確率」造成數字過度樂觀。

## 一、核心流程

| 資料 | 檔名 | 用途 |
|---|---|---|
| 訓練資料 | `data/ciciot2023_train.csv` | 只用來訓練 XGBoost 與 Isolation Forest |
| 驗證資料 | `data/ciciot2023_val.csv` | 只用來掃描並選擇 threshold |
| 最終測試資料 | `data/ciciot2023_test.csv` | 只用來做最後測試，不再調整模型或 threshold |

## 二、建議使用的 CICIoT2023 CSV

你原本資料夾有很多 `part-xxxxx.csv`。建議先用：

```text
part-00092-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv -> data/ciciot2023_train.csv
part-00037-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv -> data/ciciot2023_val.csv
part-00014-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv -> data/ciciot2023_test.csv
```

如果資料集在預設路徑，可直接執行：

```cmd
00_prepare_data_from_local.cmd
```

它會自動複製並改名。

## 三、執行順序

```cmd
01_train.cmd
02_select_threshold.cmd
03_final_test.cmd
04_run_app.cmd
```

或手動執行：

```cmd
python -m pip install -r requirements.txt
python train_model.py --data data/ciciot2023_train.csv --target label
python select_threshold.py --data data/ciciot2023_val.csv --target label
python evaluate_test.py --data data/ciciot2023_test.csv --target label
streamlit run app.py
```

## 四、為什麼這版比較嚴謹？

舊流程常見問題是：

1. 同一份資料同時拿來選 threshold 與報告準確率。
2. 資料中 Attack 太多，模型容易只猜 Attack。
3. Accuracy 很高，但 Normal recall 很低，造成實際誤報太多。

本版本修正：

1. 訓練、驗證、測試三份 CSV 分開。
2. 訓練階段採 Normal:Attack 平衡抽樣。
3. Threshold 只由 validation 資料決定。
4. Final test 只做一次最終評估，不再調 threshold。
5. Streamlit 展示模擬正常流量為主、少量攻擊穿插，比較像真實環境。

## 五、報告可用描述

本系統採用 CICIoT2023 資料集，並將不同 CSV 切片分為訓練資料、驗證資料與最終測試資料。訓練資料只用於建立 XGBoost 分類模型與 Isolation Forest 正常基準線；驗證資料用於掃描不同 threshold，選擇較能兼顧誤報與漏報的告警門檻；最終測試資料則完全不參與訓練與門檻選擇，只用於評估模型最後表現。此流程可避免使用測試資料調整參數後再回報測試準確率，使結果更具可信度。

## 六、Git 注意事項

大型 CSV 與模型檔已在 `.gitignore` 中排除，不會不小心推上 GitHub：

```text
data/*.csv
models/*.joblib
reports/*.csv
reports/*.txt
```

GitHub 上可保留程式碼、README、`.cmd` 腳本與資料夾結構。
