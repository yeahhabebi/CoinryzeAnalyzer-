# app.py — Coinryze Analyzer (Advanced Markov + Fixed CSV types + Dropdown UI)
import streamlit as st
import pandas as pd
import numpy as np
import boto3, json, datetime
from io import BytesIO
from botocore.exceptions import ClientError
from collections import defaultdict

# ---------------------- #
# ✅ Cloudflare R2 setup (via Streamlit Secrets)
# ---------------------- #
R2_KEY_ID = st.secrets["r2"]["R2_KEY_ID"]
R2_SECRET = st.secrets["r2"]["R2_SECRET"]
R2_BUCKET = st.secrets["r2"]["R2_BUCKET"]
R2_ENDPOINT = st.secrets["r2"]["R2_ENDPOINT"]

s3 = boto3.client(
    "s3",
    region_name="auto",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET,
)

F_LAST = "last_draws.csv"
F_PRED = "predictions.csv"
F_ACC = "accuracy_log.csv"

# ---------------------- #
# Utility: R2 I/O
# ---------------------- #
def r2_get_csv(key, cols):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        df = pd.read_csv(obj["Body"], dtype=str)
        for c in df.columns:
            df[c] = df[c].astype(str)
        for col in ["pick_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
        return df
    except Exception:
        return pd.DataFrame(columns=cols)

def r2_put_csv(key, df):
    try:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=csv_bytes, ContentType="text/csv")
    except ClientError as e:
        st.error(f"R2 upload error: {e}")

# ---------------------- #
# Helpers
# ---------------------- #
def infer_size(num):
    try:
        n = int(num)
        return "Small" if n <= 4 else "Big"
    except: return ""

def infer_color(num):
    try:
        n = int(num)
        if n == 0: return "Red-purple"
        if n == 5: return "Green-purple"
        return "Red" if n <= 4 else "Green"
    except: return ""

# ---------------------- #
# Load datasets
# ---------------------- #
DRAW_COLS = ["issue_id","timestamp","number","color","size"]
PRED_COLS = ["created_at","prediction","pick_count"]
ACC_COLS = ["timestamp","predicted","actual","accuracy_pct"]

last_draws = r2_get_csv(F_LAST, DRAW_COLS)
predictions = r2_get_csv(F_PRED, PRED_COLS)
accuracy = r2_get_csv(F_ACC, ACC_COLS)

# ---------------------- #
# Advanced Prediction (Hybrid Markov + Frequency)
# ---------------------- #
def advanced_predict(df, pick_count=1):
    if df.shape[0] < 3:
        return np.random.choice(range(10), size=pick_count, replace=False).tolist()

    seq = [int(x) for x in df["number"].astype(str) if x.isdigit()]
    # Build transition matrix
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(seq) - 1):
        trans[seq[i]][seq[i+1]] += 1

    last_num = seq[-1]
    next_probs = trans[last_num]
    if not next_probs:
        # fallback to frequency
        freq = pd.Series(seq).value_counts(normalize=True)
        return freq.nlargest(pick_count).index.tolist()

    total = sum(next_probs.values())
    probs = {n: c/total for n, c in next_probs.items()}
    # Combine with overall frequency (weighted hybrid)
    freq = pd.Series(seq).value_counts(normalize=True)
    hybrid_score = {n: probs.get(n, 0)*0.7 + freq.get(n, 0)*0.3 for n in range(10)}
    picks = sorted(hybrid_score, key=hybrid_score.get, reverse=True)[:pick_count]
    return picks

# ---------------------- #
# Streamlit UI
# ---------------------- #
st.set_page_config(page_title="Coinryze Analyzer (Advanced)", layout="wide")
st.title("🎯 Coinryze Analyzer — Smarter Markov Predictions")

st.sidebar.header("Settings")
pick_count = st.sidebar.slider("Prediction size", 1, 5, 2)
auto_predict = st.sidebar.checkbox("Auto-predict after adding", value=True)
st.sidebar.write("All CSVs sync automatically with Cloudflare R2")

# ---------------------- #
# Manual Entry
# ---------------------- #
st.header("Manual Input — Add Single Last Draw")

with st.form("add_draw"):
    c1,c2,c3,c4,c5 = st.columns([1.3,1.6,1,1,1])
    with c1:
        issue_id = st.text_input("Issue ID", placeholder="202510150001")
    with c2:
        timestamp = st.text_input("Timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("Number", [str(i) for i in range(10)])
    with c4:
        color = st.selectbox("Color", ["", "Red","Green","Red-purple","Green-purple"])
    with c5:
        size = st.selectbox("Size", ["", "Small","Big"])
    submit_btn = st.form_submit_button("➕ Add Draw")

if submit_btn:
    if not color: color = infer_color(number)
    if not size: size = infer_size(number)
    new = pd.DataFrame([{
        "issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "timestamp": timestamp,
        "number": number,
        "color": color,
        "size": size,
    }])
    last_draws = pd.concat([new, last_draws], ignore_index=True)
    r2_put_csv(F_LAST, last_draws)
    st.success("✅ Draw added and synced to R2")
    if auto_predict:
        picks = advanced_predict(last_draws, pick_count)
        rec = pd.DataFrame([{
            "created_at": datetime.datetime.utcnow().isoformat(),
            "prediction": json.dumps(picks),
            "pick_count": pick_count,
        }])
        predictions = pd.concat([rec, predictions], ignore_index=True)
        r2_put_csv(F_PRED, predictions)
        st.info(f"Auto-predicted next numbers: {picks}")
    st.rerun()

# ---------------------- #
# Auto-map pasted CSV
# ---------------------- #
st.subheader("📋 Paste Draw Lines (auto-map columns)")
pasted = st.text_area("Paste rows (any CSV-like format)", height=120)
if st.button("Auto-map and Add"):
    try:
        df_new = pd.read_csv(BytesIO(pasted.encode("utf-8")), sep=",")
        df_new.columns = [c.lower().strip() for c in df_new.columns]
        mapping = {c: next((m for m in DRAW_COLS if m in c), None) for c in df_new.columns}
        df_new = df_new.rename(columns=mapping)
        for c in DRAW_COLS:
            if c not in df_new.columns:
                df_new[c] = ""
        for idx, row in df_new.iterrows():
            if not row["color"]: df_new.at[idx,"color"] = infer_color(row["number"])
            if not row["size"]: df_new.at[idx,"size"] = infer_size(row["number"])
        last_draws = pd.concat([df_new[DRAW_COLS], last_draws], ignore_index=True)
        r2_put_csv(F_LAST, last_draws)
        st.success("✅ Auto-mapped and added pasted draws")
        if auto_predict:
            picks = advanced_predict(last_draws, pick_count)
            rec = pd.DataFrame([{
                "created_at": datetime.datetime.utcnow().isoformat(),
                "prediction": json.dumps(picks),
                "pick_count": pick_count,
            }])
            predictions = pd.concat([rec, predictions], ignore_index=True)
            r2_put_csv(F_PRED, predictions)
            st.info(f"Predicted: {picks}")
        st.rerun()
    except Exception as e:
        st.error(f"Could not parse: {e}")

# ---------------------- #
# Predictions + Evaluation
# ---------------------- #
st.header("📊 Smart Predictions")
if last_draws.shape[0] > 2:
    picks = advanced_predict(last_draws, pick_count)
    st.metric("Next predicted numbers", ", ".join(map(str, picks)))
    colors = [infer_color(p) for p in picks]
    st.write(pd.DataFrame({"Number": picks, "Predicted Color": colors}))
    if st.button("💾 Save Prediction"):
        rec = pd.DataFrame([{
            "created_at": datetime.datetime.utcnow().isoformat(),
            "prediction": json.dumps(picks),
            "pick_count": pick_count,
        }])
        predictions = pd.concat([rec, predictions], ignore_index=True)
        r2_put_csv(F_PRED, predictions)
        st.success("Saved to predictions.csv")
else:
    st.info("Add at least 3 draws to enable Markov predictions.")

st.header("🎯 Evaluate Last Prediction vs Latest Draw")
if not predictions.empty and not last_draws.empty:
    last_pred = json.loads(predictions.iloc[0]["prediction"])
    actual_num = int(last_draws.iloc[0]["number"])
    accuracy_pct = (actual_num in last_pred) * 100.0
    st.write(f"Last prediction: {last_pred}")
    st.write(f"Latest actual draw: {actual_num}")
    st.metric("Accuracy %", f"{accuracy_pct:.1f}%")
    if st.button("Log Accuracy"):
        rec = pd.DataFrame([{
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "predicted": json.dumps(last_pred),
            "actual": json.dumps([actual_num]),
            "accuracy_pct": accuracy_pct,
        }])
        accuracy = pd.concat([rec, accuracy], ignore_index=True)
        r2_put_csv(F_ACC, accuracy)
        st.success("Accuracy logged to R2")

st.header("📂 R2 Bucket Files")
for k in [F_LAST, F_PRED, F_ACC]:
    st.write("-", k)
    data = r2_get_csv(k, [])
    if not data.empty:
        st.download_button(f"Download {k}", data=data.to_csv(index=False).encode("utf-8"), file_name=k)

