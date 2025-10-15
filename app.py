# app.py
"""
Coinryze Analyzer — Ensemble Upgrade
Improved ensemble predictors for numbers + color + size.

Copy-paste this file as app.py and run:
  streamlit run app.py

Secrets:
Provide .streamlit/secrets.toml with:
R2_KEY_ID = "..."
R2_SECRET = "..."
R2_BUCKET = "..."
R2_ENDPOINT = "https://<account_id>.r2.cloudflarestorage.com"
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime
import json
import math

# -------------------------
# Load R2 credentials from secrets (preferred), fallback to inline values for convenience
# -------------------------
def load_r2():
    try:
        return {
            "key": st.secrets["R2_KEY_ID"],
            "secret": st.secrets["R2_SECRET"],
            "bucket": st.secrets["R2_BUCKET"],
            "endpoint": st.secrets["R2_ENDPOINT"],
        }
    except Exception:
        return {
            "key": "7423969d6d623afd9ae23258a6cd2839",
            "secret": "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c",
            "bucket": "coinryze-analyzer",
            "endpoint": "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com",
        }

R2 = load_r2()
R2_KEY = R2["key"]
R2_SECRET = R2["secret"]
R2_BUCKET = R2["bucket"]
R2_ENDPOINT = R2["endpoint"]

# filenames used
F_LAST = "last_draws.csv"
F_PRED = "predictions.csv"
F_ACC = "accuracy_log.csv"

DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

# -------------------------
# R2 helper functions
# -------------------------
def r2_client():
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY,
        aws_secret_access_key=R2_SECRET,
    )

def r2_get_bytes(key):
    try:
        s3 = r2_client()
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except Exception:
        return None

def r2_put_bytes(key, bts, content_type="text/csv"):
    try:
        s3 = r2_client()
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=bts, ContentType=content_type)
        return True
    except Exception as e:
        st.error(f"R2 upload error: {e}")
        return False

def r2_list_keys(prefix=""):
    try:
        s3 = r2_client()
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except Exception:
        return []

# -------------------------
# Load existing CSVs into session_state (persist during session)
# -------------------------
def load_df_from_r2(key, local_path=None, cols=None):
    b = r2_get_bytes(key)
    if b is not None:
        try:
            return pd.read_csv(BytesIO(b), dtype=str)
        except Exception:
            return pd.read_csv(StringIO(b.decode("utf-8")), dtype=str)
    if local_path:
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            return None
    return pd.DataFrame(columns=cols if cols else [])

if "last_draws" not in st.session_state:
    st.session_state.last_draws = load_df_from_r2(F_LAST, local_path="backend/data/seed.csv", cols=DRAW_COLS)
    for c in DRAW_COLS:
        if c not in st.session_state.last_draws.columns:
            st.session_state.last_draws[c] = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = load_df_from_r2(F_PRED, cols=["created_at","prediction","pick_count","algo","method"])
    for c in ["created_at","prediction","pick_count","algo","method"]:
        if c not in st.session_state.predictions.columns:
            st.session_state.predictions[c] = ""

if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_df_from_r2(F_ACC, cols=["timestamp","predicted","actual","accuracy_pct"])
    for c in ["timestamp","predicted","actual","accuracy_pct"]:
        if c not in st.session_state.accuracy_log.columns:
            st.session_state.accuracy_log[c] = ""

# -------------------------
# Utilities: parsing & normalization
# -------------------------
def try_cast_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def parse_line_to_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    while len(parts) < 5:
        parts.append("")
    issue_id = parts[0] or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    timestamp = parts[1] or datetime.datetime.utcnow().isoformat()
    num = try_cast_int(parts[2])
    if num is None or num < 0 or num > 9:
        return None
    color = parts[3] or infer_color(num)
    size = parts[4] or infer_size(num)
    return {"issue_id": str(issue_id), "timestamp": str(timestamp), "number": str(num), "color": color, "size": size}

def auto_map_and_normalize_df(df_in):
    # Simple heuristic mapping to DRAW_COLS; returns normalized DataFrame
    cols = list(df_in.columns)
    lower = [c.lower() for c in cols]
    mapping = {}
    for c in DRAW_COLS:
        if c in lower:
            mapping[cols[lower.index(c)]] = c
            continue
        # heuristics
        if c == "issue_id":
            candidates = [col for col in cols if "issue" in col.lower() or "id" in col.lower()]
        elif c == "timestamp":
            candidates = [col for col in cols if "time" in col.lower() or "date" in col.lower()]
        elif c == "number":
            candidates = [col for col in cols if "num" in col.lower() or col.lower() in ["n","value","number"]]
        elif c == "color":
            candidates = [col for col in cols if "color" in col.lower() or "colour" in col.lower()]
        elif c == "size":
            candidates = [col for col in cols if "size" in col.lower() or "big" in col.lower() or "small" in col.lower()]
        else:
            candidates = []
        if candidates:
            mapping[candidates[0]] = c
    # build normalized df
    norm = pd.DataFrame()
    for orig, canon in mapping.items():
        norm[canon] = df_in[orig].astype(str)
    for c in DRAW_COLS:
        if c not in norm.columns:
            norm[c] = ""
    # normalize number to digits
    norm["number"] = norm["number"].apply(lambda x: try_cast_int(x) if x is not None and str(x).strip() != "" else None)
    norm = norm.dropna(subset=["number"])
    norm["number"] = norm["number"].astype(int).astype(str)
    return norm[DRAW_COLS]

# -------------------------
# Inference helpers for color/size
# -------------------------
def infer_size(n):
    try:
        n = int(n)
    except:
        return ""
    return "Small" if 0 <= n <= 4 else "Big"

def infer_color(n):
    try:
        n = int(n)
    except:
        return ""
    if n == 0:
        return "Red-purple"
    if n == 5:
        return "Green-purple"
    return "Red" if 0 <= n <= 4 else "Green"

def compute_color_size_conditionals(df):
    # return: color_cond[num] = {color:prob,...}, size_cond likewise
    color_map = {i: {} for i in range(10)}
    size_map = {i: {} for i in range(10)}
    for _, row in df.iterrows():
        n = try_cast_int(row.get("number", None))
        if n is None: 
            continue
        c = str(row.get("color","")).strip() or infer_color(n)
        s = str(row.get("size","")).strip() or infer_size(n)
        color_map[n][c] = color_map[n].get(c,0) + 1
        size_map[n][s] = size_map[n].get(s,0) + 1
    # smoothing
    for n in range(10):
        cm = color_map[n]
        sm = size_map[n]
        if cm:
            total = sum(cm.values()) + len(cm)
            color_map[n] = {k: (v+1)/total for k,v in cm.items()}
        else:
            color_map[n] = {col: 1/4 for col in ["Red","Green","Red-purple","Green-purple"]}
        if sm:
            total = sum(sm.values()) + len(sm)
            size_map[n] = {k:(v+1)/total for k,v in sm.items()}
        else:
            size_map[n] = {"Small":0.5,"Big":0.5}
    return color_map, size_map

# -------------------------
# Ensemble predictors
# - Markov with add-one smoothing (probabilities)
# - Frequency with exponential time decay (recent draws weighted higher)
# - Streak boost: if a number has been repeating in immediate history, apply boost
# -------------------------
def build_markov_probs(nums):
    K = 10
    trans = np.ones((K,K), dtype=float)  # add-one smoothing
    for a,b in zip(nums[:-1], nums[1:]):
        trans[a,b] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    probs = trans / row_sums
    return probs  # shape KxK

def frequency_with_decay(nums, half_life=20):
    # exponential decay: weight for position i (0 most recent at end) -> exp(-ln2 * age/half_life)
    # we treat latest element as age 0
    weights = {}
    N = len(nums)
    for idx, val in enumerate(nums):
        age = N - 1 - idx  # 0 for most recent
        w = math.exp(-math.log(2) * (age / (half_life if half_life>0 else 1)))
        weights[val] = weights.get(val, 0.0) + w
    total = sum(weights.values()) if weights else 1.0
    return {i: weights.get(i, 0.0)/total for i in range(10)}

def compute_streaks(nums, max_len=5):
    # return streak_len for most recent repeating number
    if not nums:
        return {}
    streaks = {}
    # count last run lengths for each number
    for n in set(nums):
        streaks[n] = 0
    if len(nums) == 0:
        return streaks
    # check last run
    last = nums[-1]
    run = 1
    for i in range(len(nums)-2, -1, -1):
        if nums[i] == last:
            run += 1
            if run >= max_len:
                break
        else:
            break
    streaks[last] = run
    return streaks

def ensemble_predict(df, pick_count=1, w_markov=0.6, w_freq=0.3, w_recency=0.1, half_life=30, streak_boost=0.0, deterministic=True, temperature=1.0):
    """
    df: DataFrame with 'number' column (chronological order: oldest->newest preferred)
    returns: list of pick_count ints
    """
    if df is None or df.shape[0] == 0:
        return []
    # build nums list oldest->newest
    nums = [try_cast_int(x) for x in df["number"].tolist() if try_cast_int(x) is not None]
    if not nums:
        return []
    # markov
    if len(nums) >= 2:
        probs_mat = build_markov_probs(nums)
        last = nums[-1]
        markov_dist = probs_mat[last]  # probability vector length 10
    else:
        markov_dist = np.ones(10) / 10.0
    # frequency with decay
    freq_decay = frequency_with_decay(nums, half_life=half_life)  # dict 0..9
    freq_vec = np.array([freq_decay.get(i,0.0) for i in range(10)])
    # recency: produce an additional small boost for numbers seen very recently (inverse age)
    recency_scores = np.zeros(10)
    ages = {}
    # compute last index for each number (0 = oldest)
    for idx, val in enumerate(nums):
        ages[val] = idx
    N = len(nums)
    for i in range(10):
        if i in ages:
            age = N - 1 - ages[i]
            recency_scores[i] = 1.0 / (1.0 + age)
        else:
            recency_scores[i] = 0.0
    recency_vec = recency_scores / (recency_scores.sum() if recency_scores.sum()>0 else 1.0)

    # base combined
    combined = w_markov * np.array(markov_dist) + w_freq * freq_vec + w_recency * recency_vec

    # streak boost: if last number has streak>1, optionally boost it
    streaks = compute_streaks(nums)
    for n, length in streaks.items():
        if length > 1 and streak_boost != 0:
            combined[n] += streak_boost * (length - 1)

    # ensure nonnegative and normalize
    combined = np.clip(combined, 0.0, None)
    if combined.sum() <= 0:
        combined = np.ones_like(combined) / len(combined)
    else:
        combined = combined / combined.sum()

    # apply temperature
    if temperature != 1.0 and temperature > 0:
        logits = np.log(np.clip(combined, 1e-12, None)) / temperature
        exp = np.exp(logits - np.max(logits))
        combined = exp / exp.sum()

    # deterministic vs stochastic picks
    picks = []
    if deterministic:
        picks = list(np.argsort(-combined)[:pick_count])
    else:
        probs = combined.copy()
        for _ in range(pick_count):
            if probs.sum() <= 0:
                probs = np.ones_like(probs)
            pnorm = probs / probs.sum()
            choice = np.random.choice(np.arange(10), p=pnorm)
            picks.append(int(choice))
            probs[choice] = 0.0
    return [int(x) for x in picks]

# -------------------------
# Color/size selection from conditional P(color|number), P(size|number)
# -------------------------
def pick_color_size(pred_nums, df_history, deterministic=True):
    color_cond, size_cond = compute_color_size_conditionals(df_history)
    result = []
    for n in pred_nums:
        cmap = color_cond.get(n, {"Red":0.25,"Green":0.25,"Red-purple":0.25,"Green-purple":0.25})
        smap = size_cond.get(n, {"Small":0.5,"Big":0.5})
        if deterministic:
            color = max(cmap.items(), key=lambda x: x[1])[0]
            size = max(smap.items(), key=lambda x: x[1])[0]
        else:
            cols, probs = zip(*list(cmap.items()))
            color = np.random.choice(cols, p=np.array(probs)/sum(probs))
            sizes, s_probs = zip(*list(smap.items()))
            size = np.random.choice(sizes, p=np.array(s_probs)/sum(s_probs))
        result.append({"number": int(n), "color": color, "size": size})
    return result

# -------------------------
# UI: layout + controls
# -------------------------
st.set_page_config(page_title="Coinryze Analyzer — Ensemble Predict", layout="wide")
st.title("Coinryze Analyzer — Ensemble Prediction Upgrade")

# Sidebar controls for tuning
st.sidebar.header("Prediction tuning")
pick_count = st.sidebar.slider("Numbers to predict", min_value=1, max_value=5, value=1)
w_markov = st.sidebar.slider("Weight: Markov", 0.0, 1.0, 0.6)
w_freq = st.sidebar.slider("Weight: Frequency(decay)", 0.0, 1.0, 0.25)
w_recency = st.sidebar.slider("Weight: Recency", 0.0, 1.0, 0.15)
half_life = st.sidebar.slider("Half-life for decay (draws)", 1, 200, 30)
streak_boost = st.sidebar.slider("Streak boost (>=0)", 0.0, 5.0, 0.5)
deterministic = st.sidebar.checkbox("Deterministic top-k (uncheck to sample)", value=True)
temperature = st.sidebar.slider("Sampling temperature (if stochastic)", 0.1, 2.0, 1.0)
auto_predict_toggle = st.sidebar.checkbox("Auto-predict when adding data", value=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Quick actions
if st.sidebar.button("Force sync CSVs to R2"):
    r2_put_bytes(F_LAST, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
    r2_put_bytes(F_PRED, st.session_state.predictions.to_csv(index=False).encode("utf-8"))
    r2_put_bytes(F_ACC, st.session_state.accuracy_log.to_csv(index=False).encode("utf-8"))
    st.sidebar.success("Synced CSVs to R2")

# Top area: manual input (dropdowns)
st.header("Manual Input — Add Single Draw (dropdowns)")
with st.form("manual_form", clear_on_submit=True):
    c1,c2,c3,c4,c5 = st.columns([1.5,2,1,1,1])
    with c1:
        issue_id = st.text_input("issue_id", placeholder="e.g. 202510141230")
    with c2:
        timestamp = st.text_input("timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("number (0–9)", list(range(10)))
    with c4:
        color = st.selectbox("color", ["Red","Green","Red-purple","Green-purple"])
    with c5:
        size = st.selectbox("size", ["Small","Big"])
    submitted = st.form_submit_button("Add draw")

if submitted:
    new = {"issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
           "timestamp": timestamp,
           "number": str(number),
           "color": color,
           "size": size}
    st.session_state.last_draws = pd.concat([pd.DataFrame([new]), st.session_state.last_draws], ignore_index=True)
    r2_put_bytes(F_LAST, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
    st.success("Added draw and synced to R2")
    # auto-predict
    if auto_predict_toggle:
        nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                                w_markov=w_markov, w_freq=w_freq, w_recency=w_recency,
                                half_life=half_life, streak_boost=streak_boost,
                                deterministic=deterministic, temperature=temperature)
        preds = pick_color_size(nums, st.session_state.last_draws, deterministic=deterministic)
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds),
               "pick_count": pick_count, "algo":"Ensemble", "method":"det" if deterministic else "stoch"}
        st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
        r2_put_bytes(F_PRED, st.session_state.predictions.to_csv(index=False).encode("utf-8"))
        st.info(f"Auto-prediction: {preds}")

# Paste area with auto-map
st.write("---")
st.subheader("Paste rows or paste CSV text (auto-map)")
pasted = st.text_area("Paste lines (issue_id,timestamp,number,color,size) or CSV content", height=140)
if st.button("Auto-map & Add pasted lines"):
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    added = 0
    # try to parse as CSV with header first
    mapped_df = None
    if len(lines) > 0 and ("," in lines[0] and any(h in lines[0].lower() for h in ["issue","timestamp","number","color","size"])):
        try:
            df_try = pd.read_csv(StringIO("\n".join(lines)), dtype=str)
            mapped_df = auto_map_and_normalize_df(df_try)
        except Exception:
            mapped_df = None
    # fallback: line-by-line parser
    for L in lines:
        parsed = parse_line_to_row(L)
        if parsed:
            st.session_state.last_draws = pd.concat([pd.DataFrame([parsed]), st.session_state.last_draws], ignore_index=True)
            added += 1
    if mapped_df is not None and not mapped_df.empty:
        # append mapped rows
        st.session_state.last_draws = pd.concat([mapped_df, st.session_state.last_draws], ignore_index=True)
        added += mapped_df.shape[0]
    if added:
        r2_put_bytes(F_LAST, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Added {added} rows and synced to R2")
        if auto_predict_toggle:
            nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                                    w_markov=w_markov, w_freq=w_freq, w_recency=w_recency,
                                    half_life=half_life, streak_boost=streak_boost,
                                    deterministic=deterministic, temperature=temperature)
            preds = pick_color_size(nums, st.session_state.last_draws, deterministic=deterministic)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds),
                   "pick_count": pick_count, "algo":"Ensemble", "method":"det" if deterministic else "stoch"}
            st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
            r2_put_bytes(F_PRED, st.session_state.predictions.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction: {preds}")
    else:
        st.warning("No valid lines found. Check format.")

# Bulk CSV uploader (auto-map)
st.write("---")
st.subheader("Bulk CSV Upload (auto-map columns)")
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    total = 0
    for f in uploaded_files:
        try:
            dfnew = pd.read_csv(f, dtype=str)
            mapped = auto_map_and_normalize_df(dfnew)
            if mapped is None or mapped.empty:
                st.warning(f"File {f.name} couldn't be auto-mapped; skipping")
                continue
            st.session_state.last_draws = pd.concat([mapped, st.session_state.last_draws], ignore_index=True)
            total += mapped.shape[0]
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if total:
        r2_put_bytes(F_LAST, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Appended {total} rows and synced to R2")
        if auto_predict_toggle:
            nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                                    w_markov=w_markov, w_freq=w_freq, w_recency=w_recency,
                                    half_life=half_life, streak_boost=streak_boost,
                                    deterministic=deterministic, temperature=temperature)
            preds = pick_color_size(nums, st.session_state.last_draws, deterministic=deterministic)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds),
                   "pick_count": pick_count, "algo":"Ensemble", "method":"det" if deterministic else "stoch"}
            st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
            r2_put_bytes(F_PRED, st.session_state.predictions.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction: {preds}")

# Show historical draws and top 50
st.write("---")
st.subheader("Historical draws (latest 50)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No draws yet")
else:
    st.dataframe(st.session_state.last_draws.head(50))
    st.download_button("Download last_draws.csv", data=st.session_state.last_draws.to_csv(index=False).encode("utf-8"), file_name="last_draws.csv")

# Smart predictions: run now
st.write("---")
st.subheader("Smart Predictions (Ensemble)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No history to predict from")
else:
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                            w_markov=w_markov, w_freq=w_freq, w_recency=w_recency,
                            half_life=half_life, streak_boost=streak_boost,
                            deterministic=deterministic, temperature=temperature)
    preds = pick_color_size(nums, st.session_state.last_draws, deterministic=deterministic)
    st.write("Predicted numbers + color + size (current ensemble):")
    st.table(pd.DataFrame(preds))
    if st.button("Save this prediction"):
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds),
               "pick_count": pick_count, "algo":"Ensemble", "method":"det" if deterministic else "stoch"}
        st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
        r2_put_bytes(F_PRED, st.session_state.predictions.to_csv(index=False).encode("utf-8"))
        st.success("Prediction saved and synced to R2")

# Show latest saved prediction (do not regenerate; show saved)
st.write("---")
st.subheader("Latest saved prediction")
if st.session_state.predictions.shape[0] == 0:
    st.info("No saved predictions")
else:
    latest = st.session_state.predictions.iloc[0]
    try:
        latest_data = json.loads(latest["prediction"])
    except Exception:
        latest_data = []
    st.write("Saved at:", latest.get("created_at"))
    st.table(pd.DataFrame(latest_data))

# Evaluate last saved vs latest actual
st.write("---")
st.subheader("Evaluate last saved prediction vs latest actual draw")
if st.session_state.predictions.shape[0] == 0 or st.session_state.last_draws.shape[0] == 0:
    st.info("Need at least one saved prediction and one draw")
else:
    saved_pred = json.loads(st.session_state.predictions.iloc[0]["prediction"])
    latest_actual = st.session_state.last_draws.iloc[0]
    actual_num = try_cast_int(latest_actual.get("number", None))
    st.write("Latest actual:", latest_actual.to_dict())
    st.write("Saved prediction:", saved_pred)
    if actual_num is None:
        st.warning("Latest actual draw number invalid")
    else:
        saved_nums = [int(x["number"]) for x in saved_pred if "number" in x]
        overlap = len(set(saved_nums).intersection({actual_num})) / max(1, len(saved_nums)) * 100.0
        st.metric("Overlap % (numbers)", f"{overlap:.1f}%")
        if st.button("Log accuracy of saved prediction"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(saved_pred), "actual": json.dumps([actual_num]), "accuracy_pct": overlap}
            st.session_state.accuracy_log = pd.concat([pd.DataFrame([rec]), st.session_state.accuracy_log], ignore_index=True)
            r2_put_bytes(F_ACC, st.session_state.accuracy_log.to_csv(index=False).encode("utf-8"))
            st.success("Logged accuracy to accuracy_log.csv")

# Show prediction history and accuracy trend
st.write("---")
st.subheader("Predictions history (top 30)")
if st.session_state.predictions.shape[0] == 0:
    st.info("No predictions saved")
else:
    st.dataframe(st.session_state.predictions.head(30))
    st.download_button("Download predictions.csv", data=st.session_state.predictions.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

st.write("---")
st.subheader("Accuracy log & trend")
if st.session_state.accuracy_log.shape[0] == 0:
    st.info("No accuracy logs yet")
else:
    df_acc = st.session_state.accuracy_log.copy()
    df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"])
    df_acc["accuracy_pct"] = pd.to_numeric(df_acc["accuracy_pct"], errors="coerce").fillna(0)
    st.line_chart(df_acc.set_index("timestamp")["accuracy_pct"])
    st.dataframe(df_acc.head(100))

# R2 file utils
st.write("---")
st.subheader("R2 Files")
try:
    keys = r2_list_keys()
    if keys:
        for k in keys:
            st.write("•", k)
            b = r2_get_bytes(k)
            if b is not None:
                st.download_button(f"Download {k}", data=BytesIO(b), file_name=k)
    else:
        st.info("No files found in bucket")
except Exception as e:
    st.error(f"Failed to list R2 keys: {e}")

# Debug info
if show_debug:
    st.write("DEBUG")
    st.write("R2 endpoint:", R2_ENDPOINT)
    st.write("rows last_draws:", st.session_state.last_draws.shape)
    st.write("rows predictions:", st.session_state.predictions.shape)
    st.write("rows accuracy:", st.session_state.accuracy_log.shape)

# Footer tips
st.markdown("""
**Notes & tips**
- Use the sidebar sliders to tune ensemble weights. Increase `w_markov` to trust transitions more, increase `w_freq` to trust recurring hotspots, or increase `streak_boost` to favor recent repeats.
- Uncheck *Deterministic* to sample stochastically — good for discovering less-likely options.
- For production, move R2 keys to `st.secrets` and remove fallbacks from the code.
- If you want, I can add an automated tuner that optimizes weights by backtesting on historical data.
""")
