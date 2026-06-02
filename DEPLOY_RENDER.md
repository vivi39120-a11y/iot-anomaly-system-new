# 新 GitHub + Render 部署流程

## 1. 建立新的 GitHub Repository
GitHub → New repository

建議名稱：

iot-anomaly-system-new

不要選 Add README、不要選 .gitignore，因為專案內已經有。

## 2. 在 VSCode 打開本資料夾後，執行

```bash
git init
git add .
git commit -m "initial deploy version"
git branch -M main
git remote add origin https://github.com/你的帳號/iot-anomaly-system-new.git
git push -u origin main
```

如果 GitHub 要求登入，用瀏覽器登入或 Personal Access Token。

## 3. Render 建立新的線上服務
Render → New → Web Service → Connect GitHub repo

選你剛剛建立的新 repo：

iot-anomaly-system-new

設定：

- Environment: Python
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Plan: Free

按 Deploy Web Service。

## 4. 注意

這份資料夾已經把必要的 CSV、模型 joblib、報告檔保留在 Git 內，Render 才能直接啟動介面。

GitHub 單檔限制是 100MB；本專案最大 CSV 約 61MB，仍可直接推。
