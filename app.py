from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL_PATH, TARGET_COL, TRAIN_CSV, VAL_CSV, TEST_CSV, SIMULATION_MODES
from dataset_utils import find_target_col, normalize_label, prepare_xy

st.set_page_config(page_title="IoT 安全專案", layout="wide")

# -----------------------------
# 1. 載入模型包：XGBoost + Isolation Forest + threshold
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_trained_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}\n請先執行 01_train.cmd、02_select_threshold.cmd")
    package = joblib.load(MODEL_PATH)
    model = package["model"]
    iso_model = package.get("isolation_forest")
    feature_names = package["feature_names"]
    threshold = package.get("threshold")
    target_col = package.get("target_col", TARGET_COL)
    if threshold is None:
        raise ValueError("模型尚未寫入 threshold。請先執行 02_select_threshold.cmd")
    return model, iso_model, feature_names, threshold, target_col

try:
    model, iso_model, trained_features, threshold, trained_target_col = load_trained_assets()
except Exception as e:
    st.error(f"讀取模型失敗：{e}")
    st.stop()

# -----------------------------
# 2. 側邊欄：維持原版介面邏輯
# -----------------------------
st.sidebar.header("控制面板")

page = st.sidebar.radio("頁面選擇", ["即時監控頁", "資料集與模型說明頁"], index=0)

SIM_SPEED = 0.5
TREND_UPDATE_EVERY = 3
EVENT_LIMIT = 20
ALERT_LIMIT = 10

source_options = {
    "最終測試資料 ciciot2023_test.csv": TEST_CSV,
    "驗證資料 ciciot2023_val.csv": VAL_CSV,
    "訓練資料 ciciot2023_train.csv": TRAIN_CSV,
}
selected_source = st.sidebar.selectbox("展示資料來源", list(source_options.keys()))
DATA_PATH = source_options[selected_source]

sim_speed = SIM_SPEED
st.sidebar.info("每筆資料固定顯示間隔：0.5 秒")
num_samples = st.sidebar.number_input("模擬筆數", min_value=5, max_value=50, value=20, step=1)
sample_size = st.sidebar.number_input("展示資料筆數", min_value=1000, max_value=10000, value=3000, step=1000)

mode = st.sidebar.selectbox("模擬情境", list(SIMULATION_MODES.keys()))
attack_ratio = SIMULATION_MODES[mode]
burst_mode = st.sidebar.checkbox("啟用攻擊爆發區間", value=True)

# -----------------------------
# 3. 載入並清理資料
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(path_str: str, sample_size: int, target_hint: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔案：{path}\n請把 CSV 放到 data 資料夾，並依 README 改名。")

    df = pd.read_csv(path)
    target_col = find_target_col(df, target_hint)

    if sample_size < len(df):
        # 為了讓 normal / attack 都盡量存在，先依 label 分層抽樣
        y = normalize_label(df[target_col])
        n_normal = min(int(sample_size * 0.35), int((y == 0).sum()))
        n_attack = min(sample_size - n_normal, int((y == 1).sum()))
        normal_part = df[y == 0].sample(n=n_normal, random_state=42) if n_normal > 0 else pd.DataFrame()
        attack_part = df[y == 1].sample(n=n_attack, random_state=43) if n_attack > 0 else pd.DataFrame()
        df = pd.concat([normal_part, attack_part], axis=0).sample(frac=1, random_state=44).reset_index(drop=True)

    df = df.reset_index(drop=True)
    display_df = df.copy()
    X, y_binary = prepare_xy(df, target_col, feature_names=trained_features)
    return display_df, X, y_binary, target_col

try:
    display_df, processed_df, y_binary_all, target_col = load_and_clean_data(str(DATA_PATH), int(sample_size), trained_target_col)
except Exception as e:
    st.error(f"資料前處理失敗：{e}")
    st.stop()

# -----------------------------
# 4. 工具函式
# -----------------------------
def get_seq_value(row, idx=None):
    for key in ["id", "pkSeqID", "seq", "Seq", "Flow ID", "flow_id"]:
        if key in row.index:
            return row[key]
    return idx if idx is not None else "N/A"

def is_normal_binary(v):
    return int(v) == 0

def is_confirmed_attack_label(label_value):
    """原始標籤不是 BenignTraffic / Normal 時，才視為已確認攻擊類型。"""
    text = str(label_value).strip().lower()
    return text not in {"benigntraffic", "benign", "normal", "0"}

def get_label_series(df):
    if target_col in df.columns:
        return df[target_col].astype(str)
    if "label" in df.columns:
        return df["label"].astype(str)
    if "Label" in df.columns:
        return df["Label"].astype(str)
    return None

def safe_float_from_row(row, candidates):
    lower_map = {str(c).lower(): c for c in row.index}
    for name in candidates:
        col = lower_map.get(name.lower())
        if col is not None:
            try:
                return float(row[col])
            except Exception:
                return None
    return None

# -----------------------------
# 5. 規則初篩（CICIoT2023 / IoT flow 通用版）
# -----------------------------
def rule_based_screening(orig_row):
    triggered_rules = []
    rule_score = 0

    flow_duration = safe_float_from_row(orig_row, ["flow_duration", "Flow Duration", "Duration", "duration", "flow dur"])
    rate = safe_float_from_row(orig_row, ["Rate", "rate", "flow_byts_s", "Flow Bytes/s", "flow_pkts_s", "Flow Packets/s"])
    total_len = safe_float_from_row(orig_row, ["Tot sum", "Tot size", "Total Length of Fwd Packet", "Total Fwd Packet", "TotLen Fwd Pkts"])
    pkt_count = safe_float_from_row(orig_row, ["Tot Fwd Pkts", "Tot Bwd Pkts", "total packets", "packet_count", "Number"])
    iat = safe_float_from_row(orig_row, ["IAT", "Flow IAT Mean", "Flow IAT Std", "flow_iat_mean"])
    fwd_pkts = safe_float_from_row(orig_row, ["Tot Fwd Pkts", "Fwd Packet Length Mean", "fwd_pkts_tot"])
    bwd_pkts = safe_float_from_row(orig_row, ["Tot Bwd Pkts", "Bwd Packet Length Mean", "bwd_pkts_tot"])

    if rate is not None and rate > 100000:
        triggered_rules.append("資料傳輸速率偏高")
        rule_score += 2
    if total_len is not None and total_len > 5000:
        triggered_rules.append("資料量異常偏高")
        rule_score += 2
    if pkt_count is not None and pkt_count > 5000:
        triggered_rules.append("封包數異常偏高")
        rule_score += 2
    if flow_duration is not None and flow_duration == 0 and rate is not None and rate > 0:
        triggered_rules.append("連線時間極短但有流量")
        rule_score += 1
    if iat is not None and iat < 0:
        triggered_rules.append("封包間隔時間異常")
        rule_score += 1
    if fwd_pkts is not None and bwd_pkts is not None and abs(fwd_pkts - bwd_pkts) > 10000:
        triggered_rules.append("雙向封包量落差過大")
        rule_score += 1

    if not triggered_rules:
        triggered_rules.append("未命中明顯規則，主要由模型分數判斷")

    return rule_score, triggered_rules

def get_risk_level(rule_score, is_attack, attack_score, iso_score=None):
    if is_attack and (attack_score >= 0.90 or rule_score >= 3):
        return "HIGH"
    if is_attack:
        return "MEDIUM"
    if rule_score >= 2:
        return "LOW"
    if iso_score is not None and not np.isnan(iso_score) and iso_score < -0.05:
        return "LOW"
    return "NORMAL"

def get_risk_badge(level):
    if level == "HIGH":
        return "🔴 HIGH"
    if level == "MEDIUM":
        return "🟠 MEDIUM"
    if level == "LOW":
        return "🟡 LOW"
    return "🟢 NORMAL"

# -----------------------------
# 6. 用原始標籤建立展示池
# -----------------------------
@st.cache_data(show_spinner=False)
def build_display_pools(y_values):
    y = pd.Series(y_values)
    normal_indices = y[y == 0].index.tolist()
    attack_indices = y[y == 1].index.tolist()
    return normal_indices, attack_indices

normal_pool, attack_pool = build_display_pools(y_binary_all.values)
if not normal_pool:
    normal_pool = display_df.index.tolist()
if not attack_pool:
    attack_pool = display_df.index.tolist()


# -----------------------------
# 7. 雙頁 Dashboard：即時監控頁 / 資料集與模型說明頁
# -----------------------------
if page == "即時監控頁":
    st.title("物聯網惡意流量偵測系統")
    st.caption("即時監控頁：狀態卡片、最新事件、中高風險警示")
    st.caption(f"模型：XGBoost + Isolation Forest ｜ 資料來源：{DATA_PATH.name}")

    summary_placeholder = st.empty()
    status_placeholder = st.empty()
    left_col, right_col = st.columns([1.5, 1])
    with left_col:
        event_placeholder = st.empty()
    with right_col:
        alert_placeholder = st.empty()
    if st.button("開始監控演示", type="primary"):
        total_count = 0
        normal_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        results_log = []
        alert_log = []
        stats_history = []
        attack_type_counter = {}

        normal_queue = normal_pool.copy()
        attack_queue = attack_pool.copy()
        random.shuffle(normal_queue)
        random.shuffle(attack_queue)
        normal_ptr = 0
        attack_ptr = 0

        if int(num_samples) >= 10:
            burst_start = max(1, int(num_samples * 0.4))
            burst_end = min(burst_start + 5, int(num_samples))
        else:
            burst_start = max(1, int(num_samples * 0.4))
            burst_end = min(burst_start + 2, int(num_samples))

        for i in range(int(num_samples)):
            if burst_mode and burst_start <= i < burst_end:
                use_attack = random.random() < 0.60
            else:
                use_attack = random.random() < attack_ratio

            if use_attack:
                if attack_ptr >= len(attack_queue):
                    random.shuffle(attack_queue)
                    attack_ptr = 0
                idx = attack_queue[attack_ptr]
                attack_ptr += 1
            else:
                if normal_ptr >= len(normal_queue):
                    random.shuffle(normal_queue)
                    normal_ptr = 0
                idx = normal_queue[normal_ptr]
                normal_ptr += 1

            row = processed_df.iloc[idx]
            orig_row = display_df.iloc[idx]
            input_df = pd.DataFrame([row.values], columns=processed_df.columns)
            attack_score = float(model.predict_proba(input_df)[:, 1][0])
            is_attack = attack_score >= threshold
            status = "Attack" if is_attack else "Normal"

            iso_score = None
            if iso_model is not None:
                try:
                    iso_score = float(iso_model.decision_function(input_df)[0])
                except Exception:
                    iso_score = None

            rule_score, triggered_rules = rule_based_screening(orig_row)
            risk_level = get_risk_level(rule_score, is_attack, attack_score, iso_score)

            total_count += 1
            if risk_level == "NORMAL":
                normal_count += 1
            elif risk_level == "LOW":
                low_count += 1
            elif risk_level == "MEDIUM":
                medium_count += 1
            elif risk_level == "HIGH":
                high_count += 1

            current_time = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%H:%M:%S")
            seq_value = get_seq_value(orig_row, idx=idx)
            true_label = "Attack" if int(y_binary_all.iloc[idx]) == 1 else "Normal"
            original_label = str(orig_row[target_col]) if target_col in orig_row.index else true_label

            # 攻擊類型排行只統計「模型判斷為 Attack 且原始標籤也確認是攻擊」的資料。
            # BenignTraffic 是正常流量，即使被模型誤報為風險，也不放進攻擊榜單。
            confirmed_attack = is_attack and is_confirmed_attack_label(original_label)
            if confirmed_attack:
                attack_type_counter[original_label] = attack_type_counter.get(original_label, 0) + 1

            event_item = {
                "時間": current_time,
                "序列號": seq_value,
                "實際類別": true_label,
                "模型判斷": status,
                "攻擊分數": round(attack_score, 4),
                "規則分數": rule_score,
                "風險等級": get_risk_badge(risk_level),
                "最終判定": "ATTACK" if is_attack else "NORMAL",
            }
            results_log.insert(0, event_item)
            results_log = results_log[:EVENT_LIMIT]

            if risk_level in ["HIGH", "MEDIUM"]:
                rule_text = ", ".join(triggered_rules[:2]) if triggered_rules else "None"
                if is_confirmed_attack_label(original_label):
                    alert_type_text = f"{risk_level} risk｜確認攻擊：{original_label}"
                else:
                    alert_type_text = f"{risk_level} risk｜疑似異常（原始標籤：{original_label}）"
                alert_log.insert(0, {
                    "時間": current_time,
                    "序列號": seq_value,
                    "告警類型": alert_type_text,
                    "命中規則": rule_text,
                })
                alert_log = alert_log[:ALERT_LIMIT]

            stats_history.append({
                "step": total_count,
                "Normal": normal_count,
                "Low": low_count,
                "Medium": medium_count,
                "High": high_count,
            })

            with summary_placeholder.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("總流量", total_count)
                c2.metric("正常", normal_count)
                c3.metric("低風險", low_count)
                c4.metric("中風險", medium_count)
                c5.metric("高風險", high_count)

            with status_placeholder.container():
                if risk_level == "HIGH":
                    st.error(f"【高風險告警】模型判斷：Attack｜原始標籤：{original_label}｜攻擊分數 {attack_score:.4f}")
                elif risk_level == "MEDIUM":
                    st.warning(f"【中風險告警】模型判斷：Attack｜原始標籤：{original_label}｜攻擊分數 {attack_score:.4f}")
                elif risk_level == "LOW":
                    st.info("偵測到輕微異常流量")
                else:
                    st.success("目前流量正常")

            with event_placeholder.container():
                st.markdown(f"### 最新事件（最多顯示 {EVENT_LIMIT} 筆）")
                st.dataframe(pd.DataFrame(results_log), use_container_width=True, hide_index=True, height=420)

            with alert_placeholder.container():
                st.markdown(f"### 中高風險警示（最多顯示 {ALERT_LIMIT} 筆）")
                if alert_log:
                    alert_df = pd.DataFrame(alert_log)
                else:
                    alert_df = pd.DataFrame([{"時間": "", "序列號": "", "告警類型": "目前尚未出現中高風險告警", "命中規則": ""}])
                st.dataframe(
                    alert_df,
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    column_config={
                        "時間": st.column_config.TextColumn("時間", width="small"),
                        "序列號": st.column_config.TextColumn("序列號", width="small"),
                        "告警類型": st.column_config.TextColumn("告警類型", width="medium"),
                        "命中規則": st.column_config.TextColumn("命中規則", width="large"),
                    },
                )

            time.sleep(sim_speed)

        # 模擬結束後，將本輪結果存到 session_state。
        # 切到「模型分析頁」時，攻擊類型統計會讀取這裡，而不是重新統計整份資料集。
        st.session_state["last_attack_type_counter"] = attack_type_counter
        st.session_state["last_stats_history"] = stats_history
        st.session_state["last_simulation_summary"] = {
            "總流量": total_count,
            "正常": normal_count,
            "低風險": low_count,
            "中風險": medium_count,
            "高風險": high_count,
        }

        # 模擬結束後才一次顯示分析結果，避免監控過程中持續重繪造成卡頓。
        st.divider()
        st.header("風險趨勢")
        chart_df = pd.DataFrame(stats_history).set_index("step")
        st.line_chart(chart_df[["Normal", "Low", "Medium", "High"]])
        st.caption("橫軸：測試筆數｜縱軸：累積事件數量")

        st.divider()
        st.header("攻擊類型統計")
        st.caption("此區只統計本輪『開始監控演示』中，模型判斷為 Attack 且原始標籤確認為攻擊的資料。")
        if attack_type_counter:
            rank_df = pd.DataFrame([
                {"攻擊類型": k, "次數": v}
                for k, v in attack_type_counter.items()
            ]).sort_values("次數", ascending=False).reset_index(drop=True)

            top_rank_df = rank_df.head(15).sort_values("次數", ascending=True)
            st.dataframe(rank_df, use_container_width=True, hide_index=True, height=360)
        else:
            st.info("本輪監控演示沒有已確認攻擊類型資料。")
    else:
        st.info("請在側邊欄設定模擬參數後，按下『開始監控演示』。")
        st.markdown("### 即時監控流程")
        st.markdown(
            "1. 訓練資料只用於訓練 XGBoost 與正常基準線模型。\n"
            "2. 驗證資料只用於選擇 threshold。\n"
            "3. 最終測試資料只用於最後評估。\n"
            "4. Dashboard 模擬即時流量，讓正常流量成為背景，再穿插異常事件。"
        )
elif page == "資料集與模型說明頁":
    st.title("資料集與模型說明頁")
    st.caption("資料集與模型說明頁：資料集流量分布、流量統計詳細資料、模型關鍵特徵")
    st.caption(f"資料來源：{DATA_PATH.name} ｜ 展示資料筆數：{len(display_df)}")

    # --- 第一層：資料集總覽 ---
    st.header("資料集流量分布")
    dist_col1, dist_col2 = st.columns([1.2, 1])

    label_series = get_label_series(display_df)
    counts = None

    with dist_col1:
        if label_series is not None:
            counts = label_series.value_counts().reset_index()
            counts.columns = ["類別", "樣本數"]
            fig_pie = px.pie(
                counts,
                values="樣本數",
                names="類別",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_layout(
                margin=dict(l=20, r=20, t=40, b=180),
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05),
                height=900,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("找不到標籤欄位")

    with dist_col2:
        st.write("#### 流量統計詳細資料")
        if label_series is not None and counts is not None and not counts.empty:
            total_samples = counts["樣本數"].sum()
            dist_df = pd.DataFrame({
                "類別": counts["類別"],
                "樣本數": counts["樣本數"],
                "百分比": counts["樣本數"].apply(lambda v: f"{(v / total_samples) * 100:.1f}%"),
            }).reset_index(drop=True)
            st.table(dist_df)
        else:
            st.info("暫無資料")

    st.divider()

    # --- 第二層：模型分析 ---
    st.header("模型關鍵特徵")
    if hasattr(model, "feature_importances_"):
        feature_name_map = {
        # 時間 / 流量變化
        "flow_duration": "連線持續多久",
        "Flow Duration": "連線持續多久",
        "Duration": "連線時間長短",
        "Rate": "資料傳送速度",
        "Srate": "來源端傳送速度",
        "Drate": "目的端接收速度",
        "IAT": "封包之間的間隔時間",
        "Flow IAT Mean": "平均傳送間隔",
        "flow_iat_mean": "平均傳送間隔",
        "Variance": "流量變化程度",
        "Covariance": "流量變化關聯",
        "Magnitude": "流量整體大小",
        "Magnitue": "流量整體大小",
        "Radius": "流量分散程度",
        "Weight": "流量權重分數",

        # 封包 / 資料量
        "Header_Length": "封包基本資訊長度",
        "Tot sum": "總資料量",
        "Tot size": "總封包大小",
        "Number": "封包數量",
        "packet_count": "封包數量",
        "total packets": "封包數量",
        "Total Length of Fwd Packet": "送出去的資料總量",
        "TotLen Fwd Pkts": "送出去的資料總量",
        "Tot Fwd Pkts": "送出去的封包數",
        "Tot Bwd Pkts": "收到的封包數",

        # 連線類型 / 通訊方式
        "Protocol Type": "使用的通訊方式",
        "HTTP": "網頁連線",
        "HTTPS": "加密網頁連線",
        "DNS": "網址查詢",
        "TCP": "穩定連線方式",
        "UDP": "快速傳輸方式",
        "ICMP": "網路測試訊息",

        # TCP 旗標 / 控制訊號
        "fin_flag_number": "連線結束訊號次數",
        "syn_flag_number": "建立連線訊號次數",
        "rst_flag_number": "連線重置訊號次數",
        "psh_flag_number": "立即傳送資料訊號次數",
        "ack_flag_number": "確認收到訊號次數",
        "urg_flag_number": "緊急資料訊號次數",

        "fin_count": "連線結束訊號次數",
        "syn_count": "建立連線訊號次數",
        "rst_count": "連線重置訊號次數",
        "psh_count": "立即傳送資料訊號次數",
        "ack_count": "確認收到訊號次數",
        "urg_count": "緊急資料訊號次數",
    }
        importances = pd.Series(model.feature_importances_, index=trained_features)
        top_n = min(12, len(importances))
        top_importances = importances.nlargest(top_n)
        chart_data = pd.DataFrame({
            "特徵名稱": [feature_name_map.get(col, col) for col in top_importances.index],
            "重要度分數": top_importances.values,
        }).sort_values("重要度分數", ascending=True)
        fig_bar = px.bar(
            chart_data,
            x="重要度分數",
            y="特徵名稱",
            orientation="h",
            text_auto=".3f",
            color="重要度分數",
            color_continuous_scale="Blues",
        )
        fig_bar.update_layout(
            margin=dict(l=200, r=20, t=20, b=20),
            yaxis={"title": ""},
            xaxis={"title": "重要度"},
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("※ 數值越高代表特徵對模型判斷『攻擊/正常』的影響力越大。")
    else:
        st.info("目前的模型不支援顯示特徵重要度。")

