# app.py
"""
Coinryze Analyzer - Upgraded predictions (probabilistic Markov + conditional color/size)
Single-file Streamlit app. Copy-paste and run.

Notes:
- Provide your R2 credentials in .streamlit/secrets.toml as shown in README or the earlier instructions,
  keys: R2_KEY_ID, R2_SECRET, R2_BUCKET, R2_ENDPOINT
- Optional: install scikit-learn to enable ML (RandomForest) fallback/training:
    pip install scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime
import json
import traceback

# Optional ML imports
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------------
# Config / Secrets
# -------------------------
# Try st.secrets first (recommended). If not present, fallback to inline defaults.
def load_r2_credentials():
    try:
        creds = {
            "key": st.secrets["R2_KEY_ID"],
            "secret": st.secrets["R2_SECRET"],
            "bucket": st.secrets["R2_BUCKET"],
            "endpoint": st.secrets["R2_ENDPOINT"],
        }
    except Exception:
        # Fallback (already provided earlier) — keep for convenience, but move to secrets for security.
        creds = {
            "key": "7423969d6d623afd9ae23258a6cd2839",
            "secret": "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c",
            "bucket": "coinryze-analyzer",
            "endpoint": "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com",
        }
    return creds

R2 = load_r2_credentials()
R2_KEY_ID = R2["key"]
R2_SECRET = R2["secret"]
R2_BUCKET = R2["bucket"]
R2_ENDPOINT = R2["endpoint"]

F_LAST_DRAWS = "last_draws.csv"
F_PREDICTIONS = "predictions.csv"
F_ACCURACY = "accuracy_log.csv"

DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

# -------------------------
# R2 client & helpers
# -------------------------
def r2_client():
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY_ID,
        aws_secret_access_key=R2_SECRET,
    )

def r2_get_bytes(key):
    try:
        s3 = r2_client()
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except ClientError:
        return None
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
# Load/Init DataFrames in session_state (so updates persist)
# -------------------------
def load_df_from_r2_or_local(key, local_path=None, cols=None):
    b = r2_get_bytes(key)
    if b is not None:
        try:
            return pd.read_csv(BytesIO(b), dtype=str)
        except Exception:
            try:
                return pd.read_csv(StringIO(b.decode("utf-8")), dtype=str)
            except Exception:
                return pd.DataFrame(columns=cols if cols else [])
    if local_path:
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            return None
    if cols:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame()

if "last_draws" not in st.session_state:
    st.session_state.last_draws = load_df_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv", cols=DRAW_COLS)
    # ensure canonical columns
    for c in DRAW_COLS:
        if c not in st.session_state.last_draws.columns:
            st.session_state.last_draws[c] = ""

if "predictions_df" not in st.session_state:
    st.session_state.predictions_df = load_df_from_r2_or_local(F_PREDICTIONS, cols=["created_at", "prediction", "pick_count", "algo", "method"])
    for c in ["created_at", "prediction", "pick_count", "algo", "method"]:
        if c not in st.session_state.predictions_df.columns:
            st.session_state.predictions_df[c] = ""

if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_df_from_r2_or_local(F_ACCURACY, cols=["timestamp", "predicted", "actual", "accuracy_pct"])
    for c in ["timestamp", "predicted", "actual", "accuracy_pct"]:
        if c not in st.session_state.accuracy_log.columns:
            st.session_state.accuracy_log[c] = ""

# store ML model in session_state if trained
if "ml_model" not in st.session_state:
    st.session_state.ml_model = None

# -------------------------
# Parsing helpers
# -------------------------
def try_cast_num(x):
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
    number_raw = parts[2]
    number = try_cast_num(number_raw)
    if number is None:
        return None
    color = parts[3] or infer_color(number)
    size = parts[4] or infer_size(number)
    return {"issue_id": str(issue_id), "timestamp": str(timestamp), "number": str(number), "color": str(color), "size": str(size)}

# -------------------------
# Inference / fallback functions
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

# -------------------------
# Advanced prediction engine
#    - Probabilistic Markov (with add-one smoothing)
#    - Combine with frequency as fallback/boost
#    - Support deterministic top-k and stochastic sampling
#    - Also compute conditional P(color|number), P(size|number)
# -------------------------
def build_transition_matrix(nums):
    # nums: list of ints 0..9
    K = 10
    trans = np.ones((K, K), dtype=float)  # add-one smoothing
    for a, b in zip(nums[:-1], nums[1:]):
        trans[a, b] += 1.0
    # normalize to probabilities rows
    row_sums = trans.sum(axis=1, keepdims=True)
    probs = trans / row_sums
    return probs  # shape KxK

def compute_frequency(nums):
    # return dict num->freq (normalized)
    if not nums:
        return {i: 0.0 for i in range(10)}
    s = pd.Series(nums).value_counts(normalize=True)
    freq = {i: float(s.get(i, 0.0)) for i in range(10)}
    return freq

def compute_color_size_conditional(df):
    # returns dicts: P(color|num) and P(size|num)
    color_map = {i: {} for i in range(10)}
    size_map = {i: {} for i in range(10)}
    for _, row in df.iterrows():
        try:
            n = int(row["number"])
        except:
            continue
        c = str(row.get("color", "")).strip() or infer_color(n)
        s = str(row.get("size", "")).strip() or infer_size(n)
        color_map[n][c] = color_map[n].get(c, 0) + 1
        size_map[n][s] = size_map[n].get(s, 0) + 1
    # normalize to probabilities with smoothing
    for n in range(10):
        cm = color_map[n]
        sm = size_map[n]
        # apply add-one smoothing over observed categories
        if cm:
            total = sum(cm.values()) + len(cm)
            for k in cm:
                cm[k] = (cm[k] + 1) / total
        else:
            # default uniform over main four colors
            cm = {col: 1/4 for col in ["Red", "Green", "Red-purple", "Green-purple"]}
        if sm:
            total = sum(sm.values()) + len(sm)
            for k in sm:
                sm[k] = (sm[k] + 1) / total
        else:
            sm = {"Small": 0.5, "Big": 0.5}
        color_map[n] = cm
        size_map[n] = sm
    return color_map, size_map

def predict_numbers_markov_conditional(df, pick_count=1, deterministic=True, temperature=1.0):
    """
    df: last_draws dataframe (most recent first)
    deterministic: if True choose top-k by combined score; if False, sample probabilistically
    temperature: >0, lower makes distribution sharper
    """
    if df is None or df.shape[0] < 2:
        # fallback to simple frequency
        nums = [try_cast_num(x) for x in df["number"].tolist() if try_cast_num(x) is not None]
        freq = compute_frequency(nums)
        sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:pick_count]]

    # ensure chronological order: oldest -> newest
    nums = [try_cast_num(x) for x in df["number"].tolist() if try_cast_num(x) is not None]
    if not nums:
        return []
    probs = build_transition_matrix(nums)
    freq = compute_frequency(nums)

    last = nums[-1]  # last observed number
    # Markov distribution for next given last
    markov_dist = probs[last]  # length 10
    # Combine markov & freq: weighted sum then normalize
    w_markov = 0.75
    w_freq = 0.25
    combined = np.array([w_markov * markov_dist[i] + w_freq * freq.get(i, 0.0) for i in range(10)])
    # temperature scaling for stochastic sampling
    if temperature != 1.0 and temperature > 0:
        logits = np.log(np.clip(combined, 1e-12, None)) / float(temperature)
        exps = np.exp(logits - np.max(logits))
        combined = exps / exps.sum()

    # deterministic top-k
    if deterministic:
        picks = list(np.argsort(-combined)[:pick_count])
        return [int(p) for p in picks]
    else:
        # stochastic sampling without replacement using combined probabilities
        picks = []
        probs_copy = combined.copy()
        for _ in range(pick_count):
            if probs_copy.sum() <= 0:
                # fallback to uniform
                probs_copy = np.ones_like(probs_copy)
            probs_norm = probs_copy / probs_copy.sum()
            choice = np.random.choice(np.arange(10), p=probs_norm)
            picks.append(int(choice))
            probs_copy[choice] = 0  # remove chosen
        return picks

def predict_color_size_for_numbers(pred_nums, color_cond, size_cond, stochastic=False):
    results = []
    for n in pred_nums:
        # choose color by highest P(color|n) or sample if stochastic
        cmap = color_cond.get(n, {})
        smap = size_cond.get(n, {})
        if stochastic:
            # sample color
            cols, probs = zip(*list(cmap.items()))
            color = np.random.choice(cols, p=np.array(probs)/sum(probs))
            sizes, s_probs = zip(*list(smap.items()))
            size = np.random.choice(sizes, p=np.array(s_probs)/sum(s_probs))
        else:
            color = max(cmap.items(), key=lambda x: x[1])[0]
            size = max(smap.items(), key=lambda x: x[1])[0]
        results.append({"number": int(n), "color": color, "size": size})
    return results

# -------------------------
# ML helper (optional)
# -------------------------
def prepare_ml_dataset(df, window=3):
    seq = [try_cast_num(x) for x in df["number"].tolist() if try_cast_num(x) is not None]
    seq = [x for x in seq if x is not None]
    if len(seq) < window + 1:
        return None, None
    X = []
    y = []
    for i in range(window, len(seq)):
        X.append(seq[i-window:i])
        y.append(seq[i])
    X = np.array(X)
    y = np.array(y)
    # one-hot encoding
    N, W = X.shape
    X_onehot = np.zeros((N, 10 * W), dtype=int)
    for i in range(N):
        for j in range(W):
            val = X[i, j]
            if 0 <= val <= 9:
                X_onehot[i, j*10 + val] = 1
    return X_onehot, y

def train_ml_model(df, window=3):
    if not SKLEARN_AVAILABLE:
        return None, None
    data = prepare_ml_dataset(df, window)
    if data is None:
        return None, None
    X, y = data
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf, clf

def predict_ml(clf, df, window=3, pick_count=1):
    if clf is None:
        return []
    seq = [try_cast_num(x) for x in df["number"].tolist() if try_cast_num(x) is not None]
    if len(seq) < window:
        return []
    last_window = seq[-window:]
    features = np.zeros((1, 10 * window), dtype=int)
    for j, val in enumerate(last_window):
        if 0 <= val <= 9:
            features[0, j*10 + val] = 1
    try:
        probs = clf.predict_proba(features)[0]
        classes = clf.classes_
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        picks = [int(c) for c, p in pairs[:pick_count]]
        return picks
    except Exception:
        try:
            pred = clf.predict(features)[0]
            return [int(pred)]
        except Exception:
            return []

# -------------------------
# UI & Actions
# -------------------------
st.set_page_config(page_title="Coinryze Analyzer — Advanced Predict", layout="wide")
st.title("Coinryze Analyzer — Advanced Predictions (Probabilistic Markov + Conditional color/size)")

# Sidebar controls
st.sidebar.header("Settings")
pick_count = st.sidebar.number_input("Prediction count", 1, 5, 1)
deterministic = st.sidebar.checkbox("Deterministic (top-k) predictions", value=True)
temperature = st.sidebar.slider("Stochastic temperature (when sampling)", 0.1, 2.0, 1.0)
use_ml = st.sidebar.checkbox("Enable ML (RandomForest) fallback", value=False)
ml_window = st.sidebar.slider("ML window size (if using ML)", 2, 6, 3)
train_ml_btn = st.sidebar.button("Train ML model now")
auto_predict_toggle = st.sidebar.checkbox("Auto-predict after adding draws", value=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)
force_sync = st.sidebar.button("Force sync CSVs to R2")

# Handle Force sync action
if force_sync:
    r2_put_bytes(F_LAST_DRAWS, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
    r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
    r2_put_bytes(F_ACCURACY, st.session_state.accuracy_log.to_csv(index=False).encode("utf-8"))
    st.success("All CSVs synced to R2")

# Manual Input with dropdowns
st.header("Manual Input — Add Single Draw (dropdowns)")
with st.form("manual_draw", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns([1.4,2.2,1,1,1])
    with c1:
        issue_id = st.text_input("issue_id", placeholder="e.g. 202510131045")
    with c2:
        timestamp = st.text_input("timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("number (0-9)", list(range(10)))
    with c4:
        color = st.selectbox("color", ["Red", "Green", "Red-purple", "Green-purple"])
    with c5:
        size = st.selectbox("size", ["Small", "Big"])
    add_sub = st.form_submit_button("Add draw")

if add_sub:
    row = {"issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
           "timestamp": timestamp,
           "number": str(number),
           "color": color,
           "size": size}
    st.session_state.last_draws = pd.concat([pd.DataFrame([row]), st.session_state.last_draws], ignore_index=True)
    r2_put_bytes(F_LAST_DRAWS, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
    st.success("Draw added and synced to R2")
    # auto-predict
    if auto_predict_toggle:
        if use_ml and SKLEARN_AVAILABLE and st.session_state.ml_model is not None:
            nums = predict_ml(st.session_state.ml_model, st.session_state.last_draws, window=ml_window, pick_count=pick_count)
        elif use_ml and SKLEARN_AVAILABLE:
            clf, _ = train_ml_model(st.session_state.last_draws, window=ml_window)
            st.session_state.ml_model = clf
            nums = predict_ml(clf, st.session_state.last_draws, window=ml_window, pick_count=pick_count)
        else:
            nums = predict_numbers_markov_conditional(st.session_state.last_draws, pick_count=pick_count,
                                                     deterministic=deterministic, temperature=temperature)
        color_cond, size_cond = compute_color_size_conditional(st.session_state.last_draws)
        preds = predict_color_size_for_numbers(nums, color_cond, size_cond, stochastic=not deterministic)
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo": "ML" if use_ml else "Markov", "method": "det" if deterministic else "stochastic"}
        st.session_state.predictions_df = pd.concat([pd.DataFrame([rec]), st.session_state.predictions_df], ignore_index=True)
        r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
        st.info(f"Auto-prediction saved: {preds}")

# Paste area with Auto-map button
st.write("---")
st.subheader("Paste lines (issue_id,timestamp,number,color,size) — Auto-map enabled")
pasted_text = st.text_area("Paste CSV lines here (one per line). You can paste rows copied from coinryze.org", height=160)
if st.button("Parse & Auto-add pasted lines"):
    lines = [l.strip() for l in pasted_text.splitlines() if l.strip()]
    added = 0
    for L in lines:
        parsed = parse_line_to_row(L)
        if parsed:
            st.session_state.last_draws = pd.concat([pd.DataFrame([parsed]), st.session_state.last_draws], ignore_index=True)
            added += 1
    if added:
        r2_put_bytes(F_LAST_DRAWS, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Added {added} pasted rows and synced to R2")
        # auto predict
        if auto_predict_toggle:
            if use_ml and SKLEARN_AVAILABLE and st.session_state.ml_model:
                nums = predict_ml(st.session_state.ml_model, st.session_state.last_draws, window=ml_window, pick_count=pick_count)
            else:
                nums = predict_numbers_markov_conditional(st.session_state.last_draws, pick_count=pick_count,
                                                         deterministic=deterministic, temperature=temperature)
            color_cond, size_cond = compute_color_size_conditional(st.session_state.last_draws)
            preds = predict_color_size_for_numbers(nums, color_cond, size_cond, stochastic=not deterministic)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo": "ML" if use_ml else "Markov", "method": "det" if deterministic else "stochastic"}
            st.session_state.predictions_df = pd.concat([pd.DataFrame([rec]), st.session_state.predictions_df], ignore_index=True)
            r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction saved: {preds}")
    else:
        st.warning("No valid rows parsed. Check format: issue_id,timestamp,number,color,size")

# Bulk file uploader (auto-map columns)
st.write("---")
st.subheader("Bulk CSV Upload (auto-map columns)")
uploaded_files = st.file_uploader("Upload CSV(s) (will attempt to auto-map columns)", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    total = 0
    for f in uploaded_files:
        try:
            dfnew = pd.read_csv(f, dtype=str)
            # attempt to auto-map columns using heuristics
            mapped = auto_map_and_normalize_df(dfnew) if 'auto_map_and_normalize_df' in globals() else None
        except Exception:
            # fallback: try to read as simple file with columns matching
            try:
                dfnew = pd.read_csv(f, dtype=str)
                # Try to pick columns that look like number and timestamp
                if set(["issue_id", "timestamp", "number"]).issubset(set([c.lower() for c in dfnew.columns])):
                    # normalize column names
                    dfnew.columns = [c.strip() for c in dfnew.columns]
                    # ensure columns present
                    mapped = dfnew[[col for col in dfnew.columns if col.lower() in ["issue_id","timestamp","number","color","size"]]].copy()
                    # rename lowercased to canonical names if necessary
                    mapped.columns = [c if c in DRAW_COLS else c for c in mapped.columns]
                else:
                    mapped = pd.DataFrame()
            except Exception:
                mapped = pd.DataFrame()
        if mapped is None or mapped.empty:
            st.warning(f"File {f.name} could not be auto-mapped — skipping")
            continue
        # ensure canonical columns
        for c in DRAW_COLS:
            if c not in mapped.columns:
                mapped[c] = ""
        # normalize number
        mapped["number"] = mapped["number"].apply(lambda x: try_cast_num(x) if pd.notna(x) else None)
        mapped = mapped.dropna(subset=["number"])
        mapped["number"] = mapped["number"].astype(int).astype(str)
        # infer missing color/size
        for idx, row in mapped.iterrows():
            n = try_cast_num(row["number"])
            if pd.isna(row.get("color","")) or row.get("color","")=="":
                mapped.at[idx, "color"] = infer_color(n)
            if pd.isna(row.get("size","")) or row.get("size","")=="":
                mapped.at[idx, "size"] = infer_size(n)
        st.session_state.last_draws = pd.concat([mapped[DRAW_COLS], st.session_state.last_draws], ignore_index=True)
        total += mapped.shape[0]
    if total:
        r2_put_bytes(F_LAST_DRAWS, st.session_state.last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Appended {total} rows and synced to R2")
        if auto_predict_toggle:
            if use_ml and SKLEARN_AVAILABLE and st.session_state.ml_model:
                nums = predict_ml(st.session_state.ml_model, st.session_state.last_draws, window=ml_window, pick_count=pick_count)
            else:
                nums = predict_numbers_markov_conditional(st.session_state.last_draws, pick_count=pick_count,
                                                         deterministic=deterministic, temperature=temperature)
            color_cond, size_cond = compute_color_size_conditional(st.session_state.last_draws)
            preds = predict_color_size_for_numbers(nums, color_cond, size_cond, stochastic=not deterministic)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo": "ML" if use_ml else "Markov", "method": "det" if deterministic else "stochastic"}
            st.session_state.predictions_df = pd.concat([pd.DataFrame([rec]), st.session_state.predictions_df], ignore_index=True)
            r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction saved: {preds}")

# Helper: auto_map function (simple heuristic)
def auto_map_and_normalize_df(df_in):
    # tries to map input df columns to DRAW_COLS
    cols = list(df_in.columns)
    lower = [c.lower() for c in cols]
    mapping = {}
    for c in DRAW_COLS:
        # exact match
        if c in lower:
            mapping[cols[lower.index(c)]] = c
            continue
        # heuristics
        if c == "issue_id":
            candidates = [col for col in cols if "issue" in col.lower() or "id" in col.lower()]
        elif c == "timestamp":
            candidates = [col for col in cols if "time" in col.lower() or "date" in col.lower()]
        elif c == "number":
            candidates = [col for col in cols if "num" in col.lower() or col.lower() in ["n","number","value"]]
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
    # normalize number into digits
    norm["number"] = norm["number"].apply(lambda x: try_cast_num(x) if pd.notna(x) and x!="" else None)
    norm = norm.dropna(subset=["number"])
    norm["number"] = norm["number"].astype(int).astype(str)
    return norm[DRAW_COLS]

# Predictions display and control
st.write("---")
st.subheader("Smart Predictions (advance)")

# Choose algorithm action: if ML requested, allow training
if train_ml_btn:
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not installed — add it to requirements to enable ML.")
    else:
        clf, model_obj = train_ml_model(st.session_state.last_draws, window=ml_window)
        if clf is not None:
            st.session_state.ml_model = clf
            st.success("ML model trained and stored in session")
        else:
            st.warning("Not enough data to train ML model (need more history)")

# produce prediction on demand
if st.button("Run Prediction Now"):
    if use_ml and SKLEARN_AVAILABLE and st.session_state.ml_model is not None:
        nums = predict_ml(st.session_state.ml_model, st.session_state.last_draws, window=ml_window, pick_count=pick_count)
        algo = "ML"
    else:
        nums = predict_numbers_markov_conditional(st.session_state.last_draws, pick_count=pick_count, deterministic=deterministic, temperature=temperature)
        algo = "Markov"
    color_cond, size_cond = compute_color_size_conditional(st.session_state.last_draws)
    preds = predict_color_size_for_numbers(nums, color_cond, size_cond, stochastic=not deterministic)
    st.session_state.predictions_df = pd.concat([pd.DataFrame([{"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo": algo, "method": "det" if deterministic else "stoch"}]), st.session_state.predictions_df], ignore_index=True)
    r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
    st.success(f"Prediction saved: {preds}")

# Show latest saved prediction (not regenerated)
st.write("Latest saved prediction (most recent):")
if st.session_state.predictions_df is None or st.session_state.predictions_df.shape[0] == 0:
    st.info("No saved predictions yet.")
else:
    latest_pred_row = st.session_state.predictions_df.iloc[0]
    try:
        latest_pred = json.loads(latest_pred_row["prediction"])
    except Exception:
        latest_pred = []
    st.write(latest_pred_row.to_dict())
    st.table(pd.DataFrame(latest_pred))

# Save button duplicates but allows user to save current displayed picks
if st.button("Save Current Prediction Again (duplicates allowed)"):
    if "latest_pred" in locals() and latest_pred:
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(latest_pred), "pick_count": pick_count, "algo": latest_pred_row.get("algo","Markov"), "method": latest_pred_row.get("method","det")}
        st.session_state.predictions_df = pd.concat([pd.DataFrame([rec]), st.session_state.predictions_df], ignore_index=True)
        r2_put_bytes(F_PREDICTIONS, st.session_state.predictions_df.to_csv(index=False).encode("utf-8"))
        st.success("Saved duplicate prediction")

# Prediction history
st.write("---")
st.subheader("Predictions history (top 30)")
if st.session_state.predictions_df.shape[0] == 0:
    st.info("No predictions saved.")
else:
    st.dataframe(st.session_state.predictions_df.head(30))
    st.download_button("Download predictions.csv", data=st.session_state.predictions_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# Evaluate last prediction vs latest actual draw (this compares the saved prediction block to the saved latest draw)
st.write("---")
st.subheader("Evaluate Last Saved Prediction vs Latest Actual Draw")
if st.session_state.predictions_df.shape[0] == 0 or st.session_state.last_draws.shape[0] == 0:
    st.info("Need at least one saved prediction and one actual draw to evaluate.")
else:
    # latest saved prediction (most recent saved)
    saved_pred = json.loads(st.session_state.predictions_df.iloc[0]["prediction"])
    # latest actual draw row (most recent)
    latest_actual_row = st.session_state.last_draws.iloc[0]
    try:
        actual_num = try_cast_num(latest_actual_row["number"])
    except Exception:
        actual_num = None
    st.write("Saved prediction (numbers/colors/sizes):")
    st.table(pd.DataFrame(saved_pred))
    st.write("Latest actual draw:", latest_actual_row.to_dict())
    if actual_num is None:
        st.warning("Latest actual draw number is invalid.")
    else:
        # compute overlap: treat saved_pred as list of dicts with 'number'
        saved_nums = [int(item["number"]) for item in saved_pred if "number" in item]
        overlap_pct = (len(set(saved_nums).intersection({actual_num})) / max(1, len(saved_nums))) * 100.0
        st.metric("Overlap with latest actual number", f"{overlap_pct:.1f}%")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(saved_pred), "actual": json.dumps([actual_num]), "accuracy_pct": overlap_pct}
            st.session_state.accuracy_log = pd.concat([pd.DataFrame([rec]), st.session_state.accuracy_log], ignore_index=True)
            r2_put_bytes(F_ACCURACY, st.session_state.accuracy_log.to_csv(index=False).encode("utf-8"))
            st.success("Accuracy logged and synced to R2.")

# Historical draws display
st.write("---")
st.subheader("Historical draws (latest 50)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No draws yet.")
else:
    st.dataframe(st.session_state.last_draws.head(50))
    st.download_button("Download last_draws.csv", data=st.session_state.last_draws.to_csv(index=False).encode("utf-8"), file_name="last_draws.csv")

# Accuracy trend
st.write("---")
st.subheader("Accuracy log & trend")
if st.session_state.accuracy_log.shape[0] == 0:
    st.info("No accuracy logs yet.")
else:
    try:
        df_acc = st.session_state.accuracy_log.copy()
        df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"])
        df_acc["accuracy_pct"] = pd.to_numeric(df_acc["accuracy_pct"], errors="coerce").fillna(0)
        st.line_chart(df_acc.set_index("timestamp")["accuracy_pct"])
        st.dataframe(df_acc.head(100))
    except Exception as e:
        st.error(f"Failed to render accuracy chart: {e}")

# R2 file utilities and debug
st.write("---")
st.subheader("R2 Files & Utilities")
try:
    keys = r2_list_keys()
    if keys:
        for k in keys:
            st.write("•", k)
            b = r2_get_bytes(k)
            if b is not None:
                st.download_button(f"Download {k}", data=BytesIO(b), file_name=k)
    else:
        st.info("No files found in R2 bucket (or permission issue).")
except Exception as e:
    st.error(f"Failed to list R2 keys: {e}")

if show_debug:
    st.write("DEBUG INFO")
    st.write("R2 endpoint:", R2_ENDPOINT)
    st.write("R2 bucket:", R2_BUCKET)
    st.write("rows last_draws:", st.session_state.last_draws.shape[0])
    st.write("rows predictions:", st.session_state.predictions_df.shape[0])
    st.write("rows accuracy_log:", st.session_state.accuracy_log.shape[0])

# Footer: tips
st.markdown("""
**Tips**
- Use deterministic mode to reproduce top-k picks, or stochastic sampling (uncheck deterministic and set temperature>0.5) for varied exploration.
- The model uses a probabilistic Markov chain combined with frequency to produce robust next-number probabilities.
- Color & size predictions are conditional on the predicted number using P(color|number) and P(size|number) computed from historical data.
- For ML-based prediction, add `scikit-learn` to your requirements and use the Train ML button. ML requires enough sequential history to be meaningful.
""")
