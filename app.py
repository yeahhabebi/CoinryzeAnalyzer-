# app.py
"""
Coinryze Analyzer — Final (Ensemble + Auto-Tune + Markov Heatmap)
One-file Streamlit app. Copy-paste and run.

Requirements:
pip install streamlit pandas numpy boto3 matplotlib seaborn scikit-learn

Secrets:
Create .streamlit/secrets.toml (see guide below) with R2 credentials and bucket.
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime, json, math, random, time
import matplotlib.pyplot as plt
import seaborn as sns

# Optional ParameterSampler (if sklearn available) — fallback to simple random if not
try:
    from sklearn.model_selection import ParameterSampler
    SKLEARN_PARAM_SAMPLER = True
except Exception:
    SKLEARN_PARAM_SAMPLER = False

# -------------------------
# Load R2 credentials from Streamlit secrets (recommended)
# -------------------------
def load_r2_credentials():
    # Expect keys in top-level secrets: R2_KEY_ID, R2_SECRET, R2_BUCKET, R2_ENDPOINT
    try:
        return {
            "key": st.secrets["R2_KEY_ID"],
            "secret": st.secrets["R2_SECRET"],
            "bucket": st.secrets["R2_BUCKET"],
            "endpoint": st.secrets["R2_ENDPOINT"],
        }
    except Exception as e:
        st.error("R2 credentials not found in st.secrets. Please create .streamlit/secrets.toml as documented.")
        raise

R2 = load_r2_credentials()
R2_KEY = R2["key"]
R2_SECRET = R2["secret"]
R2_BUCKET = R2["bucket"]
R2_ENDPOINT = R2["endpoint"]

# Filenames in R2
F_LAST = "last_draws.csv"
F_PRED = "predictions.csv"
F_ACC = "accuracy_log.csv"
DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

# -------------------------
# R2 client & helpers
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
        obj = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return obj["Body"].read()
    except ClientError:
        return None
    except Exception:
        return None

def r2_list_keys(prefix=""):
    try:
        s3 = r2_client()
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except Exception:
        return []

def safe_read_csv_bytes(b, cols=None):
    # Attempt to read bytes into DataFrame; coerce types safely
    try:
        df = pd.read_csv(BytesIO(b), dtype=str)
    except Exception:
        try:
            df = pd.read_csv(StringIO(b.decode("utf-8")), dtype=str)
        except Exception:
            return pd.DataFrame(columns=cols if cols else [])
    # normalize pick_count -> int if present
    if "pick_count" in df.columns:
        df["pick_count"] = pd.to_numeric(df["pick_count"], errors="coerce").fillna(1).astype(int)
    return df

def load_df_from_r2(key, cols=None, local_path=None):
    b = r2_get_bytes(key)
    if b is not None:
        return safe_read_csv_bytes(b, cols=cols)
    if local_path:
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            return pd.DataFrame(columns=cols if cols else [])
    return pd.DataFrame(columns=cols if cols else [])

def save_df_to_r2(df, key):
    # ensure pick_count integer
    df_copy = df.copy()
    if "pick_count" in df_copy.columns:
        try:
            df_copy["pick_count"] = pd.to_numeric(df_copy["pick_count"], errors="coerce").fillna(1).astype(int)
        except Exception:
            df_copy["pick_count"] = 1
    b = df_copy.to_csv(index=False).encode("utf-8")
    try:
        s3 = r2_client()
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=b, ContentType="text/csv")
        return True
    except Exception as e:
        st.error(f"R2 save error: {e}")
        return False

# -------------------------
# session state initialization
# -------------------------
if "last_draws" not in st.session_state:
    st.session_state.last_draws = load_df_from_r2(F_LAST, cols=DRAW_COLS)
    # ensure canonical columns
    for c in DRAW_COLS:
        if c not in st.session_state.last_draws.columns:
            st.session_state.last_draws[c] = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = load_df_from_r2(F_PRED, cols=["created_at","prediction","pick_count","algo","method"])
    for c in ["created_at","prediction","pick_count","algo","method"]:
        if c not in st.session_state.predictions.columns:
            st.session_state.predictions[c] = ""
    if "pick_count" in st.session_state.predictions.columns:
        st.session_state.predictions["pick_count"] = pd.to_numeric(st.session_state.predictions["pick_count"], errors="coerce").fillna(1).astype(int)

if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_df_from_r2(F_ACC, cols=["timestamp","predicted","actual","accuracy_pct"])
    for c in ["timestamp","predicted","actual","accuracy_pct"]:
        if c not in st.session_state.accuracy_log.columns:
            st.session_state.accuracy_log[c] = ""

if "best_params" not in st.session_state:
    st.session_state.best_params = {
        "w_markov": 0.6,
        "w_freq": 0.25,
        "w_recency": 0.15,
        "half_life": 30,
        "streak_boost": 0.5,
        "temperature": 1.0,
        "deterministic": True,
    }

# -------------------------
# parsing & normalization helpers
# -------------------------
def normalize_number_column(df, col="number"):
    # handle strings like "7.0", " 3 ", "08"
    if col not in df.columns:
        return df
    # coerce to numeric (float), drop NaNs, convert to int safely
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[col])
    # clip / round just in case
    df[col] = df[col].round(0).astype(int)
    # ensure 0..9
    df = df[df[col].between(0,9)]
    df[col] = df[col].astype(int).astype(str)
    return df

def try_int(x):
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

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

def infer_size(n):
    try:
        n = int(n)
    except:
        return ""
    return "Small" if 0 <= n <= 4 else "Big"

def parse_line(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    while len(parts) < 5:
        parts.append("")
    issue_id = parts[0] or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    timestamp = parts[1] or datetime.datetime.utcnow().isoformat()
    n = try_int(parts[2])
    if n is None or n < 0 or n > 9:
        return None
    color = parts[3] or infer_color(n)
    size = parts[4] or infer_size(n)
    return {"issue_id": str(issue_id), "timestamp": str(timestamp), "number": str(n), "color": color, "size": size}

# -------------------------
# prediction engine
# -------------------------
def build_markov_matrix(nums):
    K = 10
    mat = np.ones((K, K), dtype=float)  # add-one smoothing
    for a, b in zip(nums[:-1], nums[1:]):
        mat[a, b] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    probs = mat / row_sums
    return mat, probs

def freq_with_decay(nums, half_life):
    weights = {}
    N = len(nums)
    for idx, v in enumerate(nums):
        age = N - 1 - idx  # 0 most recent
        w = math.exp(-math.log(2) * (age / (half_life if half_life > 0 else 1)))
        weights[v] = weights.get(v, 0.0) + w
    total = sum(weights.values()) if weights else 1.0
    return {i: weights.get(i, 0.0) / total for i in range(10)}

def compute_color_size_conditional(df):
    color_map = {i: {} for i in range(10)}
    size_map = {i: {} for i in range(10)}
    for _, row in df.iterrows():
        n = try_int(row.get("number"))
        if n is None:
            continue
        c = str(row.get("color", "")).strip() or infer_color(n)
        s = str(row.get("size", "")).strip() or infer_size(n)
        color_map[n][c] = color_map[n].get(c, 0) + 1
        size_map[n][s] = size_map[n].get(s, 0) + 1
    for n in range(10):
        cm = color_map[n]
        sm = size_map[n]
        if cm:
            total = sum(cm.values()) + len(cm)
            color_map[n] = {k: (v + 1) / total for k, v in cm.items()}
        else:
            color_map[n] = {"Red": 0.25, "Green": 0.25, "Red-purple": 0.25, "Green-purple": 0.25}
        if sm:
            total = sum(sm.values()) + len(sm)
            size_map[n] = {k: (v + 1) / total for k, v in sm.items()}
        else:
            size_map[n] = {"Small": 0.5, "Big": 0.5}
    return color_map, size_map

def compute_last_run(nums):
    if not nums:
        return {}
    last = nums[-1]
    run = 1
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] == last:
            run += 1
        else:
            break
    return {last: run}

def ensemble_predict(df_history, pick_count=1,
                     w_markov=0.6, w_freq=0.25, w_recency=0.15,
                     half_life=30, streak_boost=0.5,
                     deterministic=True, temperature=1.0):
    # df_history: oldest->newest
    if df_history is None or df_history.shape[0] < 1:
        return []
    # normalize numbers and drop invalid rows
    hist = normalize_number_column(df_history.copy(), col="number")
    if hist.shape[0] == 0:
        return []
    nums = [int(x) for x in hist["number"].astype(int).tolist()]
    if len(nums) >= 2:
        _, markov_probs = build_markov_matrix(nums)
        last = nums[-1]
        markov_dist = markov_probs[last]
    else:
        markov_dist = np.ones(10) / 10.0
    freq_vec = np.array([freq_with_decay(nums, half_life).get(i, 0) for i in range(10)])
    # recency: inverse age normalized
    N = len(nums)
    last_indexes = {}
    for idx, v in enumerate(nums):
        last_indexes[v] = idx
    recency = np.zeros(10)
    for i in range(10):
        if i in last_indexes:
            age = N - 1 - last_indexes[i]
            recency[i] = 1.0 / (1.0 + age)
    recency_vec = recency / (recency.sum() if recency.sum() > 0 else 1.0)
    # normalize weights
    total_w = w_markov + w_freq + w_recency
    if total_w <= 1e-9:
        wm_n, wf_n, wr_n = 1.0, 0.0, 0.0
    else:
        wm_n = w_markov / total_w
        wf_n = w_freq / total_w
        wr_n = w_recency / total_w
    combined = wm_n * np.array(markov_dist) + wf_n * freq_vec + wr_n * recency_vec
    # streak boost
    streaks = compute_last_run(nums)
    for n, length in streaks.items():
        if length > 1 and streak_boost > 0:
            combined[n] += streak_boost * (length - 1)
    combined = np.clip(combined, 0.0, None)
    combined = combined / (combined.sum() if combined.sum() > 0 else 1.0)
    # temperature scaling
    if temperature != 1.0 and temperature > 0:
        logits = np.log(np.clip(combined, 1e-12, None)) / temperature
        exps = np.exp(logits - np.max(logits))
        combined = exps / exps.sum()
    # picks
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

def predict_color_size_for_numbers(pred_nums, df_history, deterministic=True):
    color_cond, size_cond = compute_color_size_conditional(df_history)
    out = []
    for n in pred_nums:
        cmap = color_cond.get(n, {"Red": 0.25, "Green": 0.25, "Red-purple": 0.25, "Green-purple": 0.25})
        smap = size_cond.get(n, {"Small": 0.5, "Big": 0.5})
        if deterministic:
            color = max(cmap.items(), key=lambda x: x[1])[0]
            size = max(smap.items(), key=lambda x: x[1])[0]
        else:
            cols, probs = zip(*list(cmap.items()))
            color = np.random.choice(cols, p=np.array(probs) / sum(probs))
            sizes, s_probs = zip(*list(smap.items()))
            size = np.random.choice(sizes, p=np.array(s_probs) / sum(s_probs))
        out.append({"number": int(n), "color": color, "size": size})
    return out

# -------------------------
# Multi-objective scoring & backtest/search
# -------------------------
def score_prediction(pred_block, actual_row, weights=(0.5, 0.25, 0.25)):
    # pred_block: list of dicts with number,color,size
    # actual_row: Series with number,color,size
    try:
        actual_num = int(actual_row.get("number"))
    except Exception:
        return 0.0
    pred_nums = [p["number"] for p in pred_block]
    num_score = 1.0 if actual_num in pred_nums else 0.0
    color_score = 1.0 if str(actual_row.get("color", "")).strip() == pred_block[0]["color"] else 0.0
    size_score = 1.0 if str(actual_row.get("size", "")).strip() == pred_block[0]["size"] else 0.0
    w_num, w_col, w_size = weights
    return w_num * num_score + w_col * color_score + w_size * size_score

def build_basic_grid():
    grid = []
    w_markov_vals = [0.2, 0.5, 0.8]
    w_freq_vals = [0.1, 0.3, 0.5]
    w_rec_vals = [0.0, 0.1, 0.2]
    half_life_vals = [10, 30, 60]
    streak_vals = [0.0, 0.5, 1.0]
    temp_vals = [0.6, 1.0, 1.4]
    for wm in w_markov_vals:
        for wf in w_freq_vals:
            for wr in w_rec_vals:
                for hl in half_life_vals:
                    for sb in streak_vals:
                        for t in temp_vals:
                            grid.append({"w_markov": wm, "w_freq": wf, "w_recency": wr, "half_life": hl, "streak_boost": sb, "temperature": t})
    return grid

def backtest_grid(df_history, pick_count=1, n_backtest=200, grid=None, scoring_weights=(0.5,0.25,0.25)):
    # df_history oldest->newest: prepare numeric
    if df_history is None or df_history.shape[0] < 12:
        return None, pd.DataFrame()
    seq_df = df_history.copy().reset_index(drop=True)
    seq_df = normalize_number_column(seq_df, col="number")
    seq_df["number"] = seq_df["number"].astype(int)
    total = seq_df.shape[0]
    start = max(5, total - n_backtest)
    idxs = list(range(start, total))
    results = []
    for params in grid:
        score_acc = 0.0
        cnt = 0
        for t in idxs:
            hist = seq_df.iloc[:t]
            actual = seq_df.iloc[t]
            nums = ensemble_predict(hist, pick_count=pick_count,
                                    w_markov=params["w_markov"],
                                    w_freq=params["w_freq"],
                                    w_recency=params["w_recency"],
                                    half_life=params["half_life"],
                                    streak_boost=params["streak_boost"],
                                    deterministic=True,
                                    temperature=params["temperature"])
            preds = predict_color_size_for_numbers(nums, hist, deterministic=True)
            s = score_prediction(preds, actual, weights=scoring_weights)
            score_acc += s
            cnt += 1
        avg = score_acc / cnt if cnt>0 else 0.0
        results.append({**params, "score": avg})
    df_res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    best = df_res.iloc[0].to_dict() if not df_res.empty else None
    return best, df_res

def randomized_search(df_history, pick_count=1, n_iter=60, n_backtest=200, scoring_weights=(0.5,0.25,0.25)):
    # parameter distribution
    param_dist = {
        "w_markov": [0.1,0.2,0.4,0.6,0.8],
        "w_freq": [0.1,0.2,0.3,0.5],
        "w_recency": [0.0,0.05,0.1,0.2],
        "half_life": [5,10,20,30,60],
        "streak_boost": [0.0,0.25,0.5,1.0],
        "temperature": [0.6,0.8,1.0,1.2,1.5]
    }
    if SKLEARN_PARAM_SAMPLER:
        sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    else:
        sampler = []
        keys = list(param_dist.keys())
        for _ in range(n_iter):
            sampler.append({k: random.choice(param_dist[k]) for k in keys})
    best_score = -1.0
    best_params = None
    results = []
    pb = st.progress(0)
    for i, params in enumerate(sampler):
        best_p, df_r = backtest_grid(df_history, pick_count=pick_count, n_backtest=n_backtest, grid=[params], scoring_weights=scoring_weights)
        score = df_r.iloc[0]["score"] if (not df_r.empty) else 0.0
        results.append({**params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params.copy()
        pb.progress(int((i+1)/len(sampler)*100))
    pb.empty()
    df_out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, df_out

# -------------------------
# Markov heatmap helpers
# -------------------------
def markov_matrices(df_history):
    seq = normalize_number_column(df_history.copy(), col="number")
    nums = [int(x) for x in seq["number"].astype(int).tolist()]
    K = 10
    counts = np.zeros((K,K), dtype=float)
    for a,b in zip(nums[:-1], nums[1:]):
        counts[a,b] += 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, row_sums, where=row_sums!=0)
    probs = np.nan_to_num(probs)
    return counts, probs

def plot_markov_heatmap(probs):
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="magma", ax=ax, cbar_kws={'label':'P(next|current)'})
    ax.set_xlabel("Next number")
    ax.set_ylabel("Current number")
    ax.set_title("Markov Transition Probabilities (P(next | current))")
    plt.tight_layout()
    return fig

# -------------------------
# UI: Streamlit layout
# -------------------------
st.set_page_config(page_title="Coinryze Analyzer — Final", layout="wide")
st.title("Coinryze Analyzer — Final (Ensemble + Auto-Tune + Markov Heatmap)")

# Sidebar controls
st.sidebar.header("Controls & Backtest")
pick_count = st.sidebar.number_input("Prediction size (numbers)", 1, 5, value=1)
deterministic = st.sidebar.checkbox("Deterministic (top-k)", value=st.session_state.best_params["deterministic"])
temperature = st.sidebar.slider("Sampling temperature (if stochastic)", 0.4, 2.0, float(st.session_state.best_params["temperature"]))
half_life_override = st.sidebar.slider("Half-life override", 5, 200, int(st.session_state.best_params["half_life"]))
streak_boost_override = st.sidebar.slider("Streak boost override", 0.0, 2.0, float(st.session_state.best_params["streak_boost"]))
n_backtest = st.sidebar.number_input("Backtest window (last N draws)", min_value=20, max_value=2000, value=200, step=10)
grid_btn = st.sidebar.button("Run grid backtest (deterministic & quick)")
rand_btn = st.sidebar.button("Run randomized auto-tune (Bayesian-like, slower)")
if st.sidebar.button("Force sync CSVs to R2"):
    save_df_to_r2(st.session_state.last_draws, F_LAST)
    save_df_to_r2(st.session_state.predictions, F_PRED)
    save_df_to_r2(st.session_state.accuracy_log, F_ACC)
    st.sidebar.success("Synced CSVs to R2")

# Manual input form
st.header("Manual Input — Add Single Draw")
with st.form("manual_draw", clear_on_submit=True):
    c1,c2,c3,c4,c5 = st.columns([1.4,2.2,1,1,1])
    with c1:
        issue_id = st.text_input("issue_id", placeholder="202510131045")
    with c2:
        timestamp = st.text_input("timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("number (0-9)", list(range(10)))
    with c4:
        color = st.selectbox("color", ["Red","Green","Red-purple","Green-purple"])
    with c5:
        size = st.selectbox("size", ["Small","Big"])
    add_sub = st.form_submit_button("Add draw")

if add_sub:
    row = {"issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
           "timestamp": timestamp, "number": str(number), "color": color, "size": size}
    st.session_state.last_draws = pd.concat([pd.DataFrame([row]), st.session_state.last_draws], ignore_index=True)
    save_df_to_r2(st.session_state.last_draws, F_LAST)
    st.success("Added draw and synced to R2")
    # optional auto-predict using best_params
    if st.sidebar.checkbox("Auto predict on add", value=True):
        bp = st.session_state.best_params
        nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                                 w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                                 half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                                 deterministic=bp["deterministic"], temperature=bp["temperature"])
        preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method":"det" if bp["deterministic"] else "stoch"}
        st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
        save_df_to_r2(st.session_state.predictions, F_PRED)
        st.info("Auto-prediction generated & saved")

# Paste / upload
st.write("---")
st.subheader("Paste lines (auto-map) or Bulk CSV Upload")
pasted = st.text_area("Paste lines from coinryze.org or CSV text (issue_id,timestamp,number,color,size)", height=140)
if st.button("Parse & Add Pasted Lines"):
    lines = [l.strip() for l in pasted.splitlines() if l.strip()]
    added = 0
    for L in lines:
        parsed = parse_line(L)
        if parsed:
            st.session_state.last_draws = pd.concat([pd.DataFrame([parsed]), st.session_state.last_draws], ignore_index=True)
            added += 1
    if added:
        save_df_to_r2(st.session_state.last_draws, F_LAST)
        st.success(f"Added {added} pasted rows and synced to R2")
    else:
        st.warning("No valid rows parsed. Check format.")

uploaded_files = st.file_uploader("Upload CSV(s) (auto-map columns)", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    total_added = 0
    for f in uploaded_files:
        try:
            df_new = pd.read_csv(f, dtype=str)
            # attempt auto-map
            def auto_map(df_in):
                cols = list(df_in.columns)
                lower = [c.lower() for c in cols]
                mapping = {}
                for c in DRAW_COLS:
                    if c in lower:
                        mapping[cols[lower.index(c)]] = c
                        continue
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
                norm = pd.DataFrame()
                for orig, canon in mapping.items():
                    norm[canon] = df_in[orig].astype(str)
                for c in DRAW_COLS:
                    if c not in norm.columns:
                        norm[c] = ""
                norm = normalize_number_column(norm, col="number")
                # infer missing color/size
                for idx, row in norm.iterrows():
                    n = try_int(row["number"])
                    if not row.get("color"):
                        norm.at[idx,"color"] = infer_color(n)
                    if not row.get("size"):
                        norm.at[idx,"size"] = infer_size(n)
                return norm[DRAW_COLS]
            mapped = auto_map(df_new)
            if mapped is not None and not mapped.empty:
                st.session_state.last_draws = pd.concat([mapped, st.session_state.last_draws], ignore_index=True)
                total_added += mapped.shape[0]
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if total_added:
        save_df_to_r2(st.session_state.last_draws, F_LAST)
        st.success(f"Appended {total_added} rows and synced to R2")

# show historical draws
st.write("---")
st.subheader("Historical draws (latest 50)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No draws yet")
else:
    st.dataframe(st.session_state.last_draws.head(50))

# Backtest / Auto-Tune actions
st.write("---")
st.subheader("Backtest & Auto-Tune (Grid + Randomized search)")
st.write("Multi-objective scoring: number overlap (0.5), color (0.25), size (0.25)")

if grid_btn:
    with st.spinner("Running grid backtest..."):
        grid = build_basic_grid()
        best, df_res = backtest_grid(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count, n_backtest=n_backtest, grid=grid)
        if best is None:
            st.warning("Not enough history to run backtest (need ~12+ rows).")
        else:
            st.session_state.best_params.update({
                "w_markov": float(best["w_markov"]),
                "w_freq": float(best["w_freq"]),
                "w_recency": float(best["w_recency"]),
                "half_life": int(best["half_life"]),
                "streak_boost": float(best["streak_boost"]),
                "temperature": float(best.get("temperature", 1.0)),
                "deterministic": True
            })
            st.success("Grid backtest complete — best params applied to session.")
            st.dataframe(df_res.head(10))
            st.download_button("Download grid results", data=df_res.to_csv(index=False).encode("utf-8"), file_name="grid_results.csv")

if rand_btn:
    with st.spinner("Running randomized Auto-Tune. This may take time..."):
        best_rand, df_rand = randomized_search(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count, n_iter=60, n_backtest=n_backtest)
        if best_rand is None:
            st.warning("Auto-tune failed (insufficient history).")
        else:
            st.session_state.best_params.update({
                "w_markov": float(best_rand["w_markov"]),
                "w_freq": float(best_rand["w_freq"]),
                "w_recency": float(best_rand["w_recency"]),
                "half_life": int(best_rand["half_life"]),
                "streak_boost": float(best_rand["streak_boost"]),
                "temperature": float(best_rand.get("temperature", 1.0)),
                "deterministic": True
            })
            st.success("Randomized Auto-Tune complete — best params applied.")
            st.dataframe(df_rand.head(10))
            st.download_button("Download randomized search results", data=df_rand.to_csv(index=False).encode("utf-8"), file_name="random_search_results.csv")

# Live prediction using applied params (with overrides)
st.write("---")
st.subheader("Current ensemble prediction (applied tuned params)")
bp = st.session_state.best_params.copy()
# apply small overrides from sidebar
bp["half_life"] = half_life_override
bp["streak_boost"] = streak_boost_override
bp["deterministic"] = deterministic
bp["temperature"] = temperature

if st.button("Show current probabilities & prediction"):
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                             w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                             half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                             deterministic=bp["deterministic"], temperature=bp["temperature"])
    preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
    st.write("Predicted block:")
    st.table(pd.DataFrame(preds))
    # markov heatmap
    counts, probs = markov_matrices(st.session_state.last_draws)
    fig = plot_markov_heatmap(probs)
    st.pyplot(fig)

# Save current prediction
if st.button("Save current prediction"):
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                             w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                             half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                             deterministic=bp["deterministic"], temperature=bp["temperature"])
    preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
    rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method":"det" if bp["deterministic"] else "stoch"}
    st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
    save_df_to_r2(st.session_state.predictions, F_PRED)
    st.success("Prediction saved and synced to R2")

# Latest saved prediction display
st.write("---")
st.subheader("Latest saved prediction (not regenerated)")
if st.session_state.predictions.shape[0] == 0:
    st.info("No saved predictions yet.")
else:
    lp = st.session_state.predictions.iloc[0]
    try:
        saved_block = json.loads(lp["prediction"])
    except Exception:
        saved_block = []
    st.write("Saved at:", lp.get("created_at"))
    st.table(pd.DataFrame(saved_block))

# Evaluate last saved prediction vs latest actual draw
st.write("---")
st.subheader("Evaluate last saved prediction vs latest actual draw")
if st.session_state.predictions.shape[0] == 0 or st.session_state.last_draws.shape[0] == 0:
    st.info("Need a saved prediction and at least one actual draw to evaluate.")
else:
    saved_block = json.loads(st.session_state.predictions.iloc[0]["prediction"])
    latest_actual = st.session_state.last_draws.iloc[0]
    actual_num = try_int(latest_actual.get("number"))
    st.write("Latest actual draw:", latest_actual.to_dict())
    st.write("Saved prediction block:", saved_block)
    if actual_num is None:
        st.warning("Latest actual number invalid.")
    else:
        saved_nums = [int(x["number"]) for x in saved_block if "number" in x]
        overlap_pct = (len(set(saved_nums).intersection({actual_num})) / max(1, len(saved_nums))) * 100.0
        st.metric("Overlap % (numbers)", f"{overlap_pct:.1f}%")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(saved_block), "actual": json.dumps([actual_num]), "accuracy_pct": overlap_pct}
            st.session_state.accuracy_log = pd.concat([pd.DataFrame([rec]), st.session_state.accuracy_log], ignore_index=True)
            save_df_to_r2(st.session_state.accuracy_log, F_ACC)
            st.success("Accuracy logged and synced to R2")

# Accuracy log & chart
st.write("---")
st.subheader("Accuracy log & trend")
if st.session_state.accuracy_log.shape[0] == 0:
    st.info("No accuracy logs yet.")
else:
    df_acc = st.session_state.accuracy_log.copy()
    df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"], errors="coerce")
    df_acc["accuracy_pct"] = pd.to_numeric(df_acc["accuracy_pct"], errors="coerce").fillna(0)
    st.line_chart(df_acc.set_index("timestamp")["accuracy_pct"])
    st.dataframe(df_acc.head(200))
    st.download_button("Download accuracy_log.csv", data=df_acc.to_csv(index=False).encode("utf-8"), file_name="accuracy_log.csv")

# R2 bucket files list
st.write("---")
st.subheader("R2 Bucket Files")
try:
    keys = r2_list_keys()
    if keys:
        for k in keys:
            st.write("-", k)
    else:
        st.info("No files found in bucket or permission issue.")
except Exception as e:
    st.error(f"Cannot list R2 keys: {e}")

# debug
if st.sidebar.checkbox("Show debug info"):
    st.write("DEBUG")
    st.write("last_draws rows:", st.session_state.last_draws.shape)
    st.write("predictions rows:", st.session_state.predictions.shape)
    st.write("accuracy rows:", st.session_state.accuracy_log.shape)
    st.json(st.session_state.best_params)

st.markdown("""
**Notes**
- Put your R2 credentials into `.streamlit/secrets.toml` (see guide after code).
- Run grid backtest first to get a quick reasonable parameter set. Use randomized auto-tune for a deeper search.
- Backtest uses combined scoring (number overlap 0.5, color 0.25, size 0.25) — you can change weights in `score_prediction`.
""")
