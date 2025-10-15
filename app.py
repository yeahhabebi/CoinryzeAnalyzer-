# app.py - Coinryze Analyzer (Final upgrade)
# One-file Streamlit app: ensemble predictions + backtest/autotune + Markov heatmap + R2 sync

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime, json, math, random, time

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# optional ParameterSampler for randomized/bayesian-like search
try:
    from sklearn.model_selection import ParameterSampler
    SKLEARN_PARAM_SAMPLER = True
except Exception:
    SKLEARN_PARAM_SAMPLER = False

# -------------------------
# R2 Credentials (from secrets.toml recommended)
# -------------------------
def load_r2_creds():
    try:
        r = st.secrets["r2"]
        return {
            "key": r.get("R2_KEY_ID") or r.get("key"),
            "secret": r.get("R2_SECRET") or r.get("secret"),
            "bucket": r.get("R2_BUCKET") or r.get("bucket"),
            "endpoint": r.get("R2_ENDPOINT") or r.get("endpoint"),
        }
    except Exception:
        # fallback defaults (convenience) - move to secrets for production
        return {
            "key": "7423969d6d623afd9ae23258a6cd2839",
            "secret": "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c",
            "bucket": "coinryze-analyzer",
            "endpoint": "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com",
        }

R2 = load_r2_creds()
R2_KEY = R2["key"]
R2_SECRET = R2["secret"]
R2_BUCKET = R2["bucket"]
R2_ENDPOINT = R2["endpoint"]

# Filenames
F_LAST = "last_draws.csv"
F_PRED = "predictions.csv"
F_ACC = "accuracy_log.csv"

DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

# -------------------------
# S3/R2 helpers
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

def read_csv_bytes_safe(b, cols=None):
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

def load_df(key, cols=None, local_path=None):
    b = r2_get_bytes(key)
    if b is not None:
        return read_csv_bytes_safe(b, cols=cols)
    if local_path:
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            return pd.DataFrame(columns=cols if cols else [])
    return pd.DataFrame(columns=cols if cols else [])

def save_df_to_r2(df, key):
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
        st.error(f"R2 upload error: {e}")
        return False

# -------------------------
# Initialize / session state
# -------------------------
if "last_draws" not in st.session_state:
    st.session_state.last_draws = load_df(F_LAST, cols=DRAW_COLS)
    for c in DRAW_COLS:
        if c not in st.session_state.last_draws.columns:
            st.session_state.last_draws[c] = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = load_df(F_PRED, cols=["created_at","prediction","pick_count","algo","method"])
    for c in ["created_at","prediction","pick_count","algo","method"]:
        if c not in st.session_state.predictions.columns:
            st.session_state.predictions[c] = ""
    if "pick_count" in st.session_state.predictions.columns:
        st.session_state.predictions["pick_count"] = pd.to_numeric(st.session_state.predictions["pick_count"], errors="coerce").fillna(1).astype(int)

if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_df(F_ACC, cols=["timestamp","predicted","actual","accuracy_pct"])
    for c in ["timestamp","predicted","actual","accuracy_pct"]:
        if c not in st.session_state.accuracy_log.columns:
            st.session_state.accuracy_log[c] = ""

if "best_params" not in st.session_state:
    # sensible defaults
    st.session_state.best_params = {
        "w_markov": 0.6,
        "w_freq": 0.25,
        "w_recency": 0.15,
        "half_life": 30,
        "streak_boost": 0.5,
        "temperature": 1.0,
        "deterministic": True
    }

# -------------------------
# Parsing, inference helpers
# -------------------------
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

def parse_line_to_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    while len(parts) < 5:
        parts.append("")
    issue_id = parts[0] or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    timestamp = parts[1] or datetime.datetime.utcnow().isoformat()
    num = try_int(parts[2])
    if num is None or num < 0 or num > 9:
        return None
    color = parts[3] or infer_color(num)
    size = parts[4] or infer_size(num)
    return {"issue_id": str(issue_id), "timestamp": str(timestamp), "number": str(num), "color": color, "size": size}

# -------------------------
# Prediction core: ensemble Markov + freq decay + recency + streak
# -------------------------
def build_markov_matrix(nums):
    K = 10
    mat = np.ones((K,K), dtype=float)  # add-one smoothing
    for a,b in zip(nums[:-1], nums[1:]):
        mat[a, b] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    probs = mat / row_sums
    return probs

def freq_with_decay(nums, half_life):
    weights = {}
    N = len(nums)
    for idx, v in enumerate(nums):
        age = N - 1 - idx  # 0 most recent
        w = math.exp(-math.log(2) * (age / (half_life if half_life>0 else 1)))
        weights[v] = weights.get(v, 0.0) + w
    total = sum(weights.values()) if weights else 1.0
    return {i: weights.get(i, 0.0)/total for i in range(10)}

def compute_color_size_conditional(df):
    color_map = {i: {} for i in range(10)}
    size_map = {i: {} for i in range(10)}
    for _, row in df.iterrows():
        n = try_int(row.get("number", None))
        if n is None:
            continue
        c = str(row.get("color", "")).strip() or infer_color(n)
        s = str(row.get("size", "")).strip() or infer_size(n)
        color_map[n][c] = color_map[n].get(c, 0) + 1
        size_map[n][s] = size_map[n].get(s, 0) + 1
    for n in range(10):
        cm = color_map[n]; sm = size_map[n]
        if cm:
            total = sum(cm.values()) + len(cm)
            color_map[n] = {k: (v+1)/total for k,v in cm.items()}
        else:
            color_map[n] = {"Red":0.25,"Green":0.25,"Red-purple":0.25,"Green-purple":0.25}
        if sm:
            total = sum(sm.values()) + len(sm)
            size_map[n] = {k:(v+1)/total for k,v in sm.items()}
        else:
            size_map[n] = {"Small":0.5,"Big":0.5}
    return color_map, size_map

def compute_streaks(nums):
    if not nums:
        return {}
    last = nums[-1]
    run = 1
    for i in range(len(nums)-2, -1, -1):
        if nums[i] == last:
            run += 1
        else:
            break
    return {last: run}

def ensemble_predict(df_history, pick_count=1,
                     w_markov=0.6, w_freq=0.25, w_recency=0.15,
                     half_life=30, streak_boost=0.5,
                     deterministic=True, temperature=1.0):
    if df_history is None or df_history.shape[0] == 0:
        return []
    nums = [try_int(x) for x in df_history["number"].tolist() if try_int(x) is not None]
    if not nums:
        return []
    if len(nums) >= 2:
        probs_mat = build_markov_matrix(nums)
        last = nums[-1]
        markov_dist = probs_mat[last]
    else:
        markov_dist = np.ones(10) / 10.0
    freq_vec = np.array([freq_with_decay(nums, half_life).get(i,0) for i in range(10)])
    N = len(nums)
    last_indexes = {}
    for idx, v in enumerate(nums):
        last_indexes[v] = idx
    recency_vec = np.zeros(10)
    for i in range(10):
        if i in last_indexes:
            age = N - 1 - last_indexes[i]
            recency_vec[i] = 1.0/(1.0 + age)
    recency_vec = recency_vec / (recency_vec.sum() if recency_vec.sum()>0 else 1.0)
    # normalize weight
    total_w = w_markov + w_freq + w_recency
    if total_w <= 0:
        w_markov_n, w_freq_n, w_recency_n = 1.0, 0.0, 0.0
    else:
        w_markov_n = w_markov / total_w
        w_freq_n = w_freq / total_w
        w_recency_n = w_recency / total_w
    combined = w_markov_n * np.array(markov_dist) + w_freq_n * freq_vec + w_recency_n * recency_vec
    streaks = compute_streaks(nums)
    for n, length in streaks.items():
        if length > 1 and streak_boost > 0:
            combined[n] += streak_boost * (length - 1)
    combined = np.clip(combined, 0.0, None)
    combined = combined / (combined.sum() if combined.sum()>0 else 1.0)
    if temperature != 1.0 and temperature > 0:
        logits = np.log(np.clip(combined, 1e-12, None)) / temperature
        exps = np.exp(logits - np.max(logits))
        combined = exps / exps.sum()
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
        out.append({"number": int(n), "color": color, "size": size})
    return out

# -------------------------
# Backtest / AutoTune (multi-objective scoring)
# -------------------------
def score_prediction_vs_actual(pred_block, actual_row, weights=(0.5, 0.25, 0.25)):
    """
    weights: (number_overlap_weight, color_weight, size_weight)
    pred_block: list of dicts [{'number':n,'color':c,'size':s},...]
    actual_row: a series/dict with 'number','color','size'
    """
    num_overlap = 1.0 if int(actual_row.get("number")) in [p["number"] for p in pred_block] else 0.0
    color_score = 1.0 if str(actual_row.get("color","")).strip() == pred_block[0]["color"] else 0.0
    size_score = 1.0 if str(actual_row.get("size","")).strip() == pred_block[0]["size"] else 0.0
    w_num, w_color, w_size = weights
    return w_num * num_overlap + w_color * color_score + w_size * size_score

def build_grid_for_search():
    # modest grid to keep runtime reasonable; we'll add random sampling for broader search
    grid = []
    w_markov_vals = [0.2, 0.5, 0.8]
    w_freq_vals = [0.1, 0.3, 0.5]
    half_life_vals = [10, 30, 60]
    streak_vals = [0.0, 0.5, 1.0]
    temp_vals = [0.6, 1.0, 1.4]
    for wm in w_markov_vals:
        for wf in w_freq_vals:
            for wr in [0.0, 0.1, 0.2]:
                for hl in half_life_vals:
                    for sb in streak_vals:
                        for t in temp_vals:
                            grid.append({"w_markov":wm, "w_freq":wf, "w_recency":wr, "half_life":hl, "streak_boost":sb, "temperature":t})
    return grid

def backtest_grid(df_history, pick_count=1, n_backtest=200, grid=None, scoring_weights=(0.5,0.25,0.25)):
    """
    Simulate sequential predictions on the last n_backtest draws, evaluate average multi-objective score.
    df_history: oldest->newest
    """
    if df_history is None or df_history.shape[0] < 12:
        return None, pd.DataFrame()
    seq_df = df_history.reset_index(drop=True).dropna(subset=["number"])
    seq_df["number"] = seq_df["number"].astype(int)
    total = seq_df.shape[0]
    start = max(5, total - n_backtest)
    indices = list(range(start, total))  # for each index t, predict using seq_df[:t] and compare to seq_df.iloc[t]
    results = []
    for params in grid:
        score_acc = 0.0
        cnt = 0
        for t in indices:
            hist = seq_df.iloc[:t]
            actual = seq_df.iloc[t]
            nums = ensemble_predict(hist, pick_count=pick_count,
                                    w_markov=params["w_markov"], w_freq=params["w_freq"], w_recency=params["w_recency"],
                                    half_life=params["half_life"], streak_boost=params["streak_boost"],
                                    deterministic=True, temperature=params["temperature"])
            preds = predict_color_size_for_numbers(nums, hist, deterministic=True)
            score = score_prediction_vs_actual(preds, actual, weights=scoring_weights)
            score_acc += score
            cnt += 1
        avg = score_acc / cnt if cnt>0 else 0.0
        row = params.copy()
        row["score"] = avg
        results.append(row)
    df_res = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    best = df_res.iloc[0].to_dict() if not df_res.empty else None
    return best, df_res

def randomized_search(df_history, pick_count=1, n_iter=60, n_backtest=200, scoring_weights=(0.5,0.25,0.25)):
    # Random/bayesian-like search using ParameterSampler if sklearn available, otherwise random sampling
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
            s = {k: random.choice(param_dist[k]) for k in keys}
            sampler.append(s)
    best_score = -1.0
    best_params = None
    results = []
    grid_count = len(sampler)
    pb = st.progress(0)
    for i, params in enumerate(sampler):
        best_grid, df_res = backtest_grid(df_history, pick_count=pick_count, n_backtest=n_backtest, grid=[params], scoring_weights=scoring_weights)
        score = df_res.iloc[0]["score"] if (not df_res.empty) else 0.0
        results.append({**params, "score": score})
        if score > best_score:
            best_score = score
            best_params = params.copy()
        pb.progress(int((i+1)/grid_count*100))
    pb.empty()
    df_out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    return best_params, df_out

# -------------------------
# Visualization: Markov Heatmap
# -------------------------
def markov_transition_matrix(df_history):
    nums = [try_int(x) for x in df_history["number"].tolist() if try_int(x) is not None]
    K = 10
    mat = np.zeros((K,K), dtype=float)
    for a,b in zip(nums[:-1], nums[1:]):
        mat[a,b] += 1
    # row-normalize
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        probs = np.divide(mat, row_sums, where=row_sums!=0)
    probs = np.nan_to_num(probs)
    return mat, probs

def plot_markov_heatmap(probs):
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(probs, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={'label':'P(next|current)'})
    ax.set_xlabel("Next number")
    ax.set_ylabel("Current number")
    ax.set_title("Markov Transition Probabilities")
    return fig

# -------------------------
# Streamlit UI Layout
# -------------------------
st.set_page_config(page_title="Coinryze Analyzer — Final", layout="wide")
st.title("Coinryze Analyzer — Final (Ensemble + Auto-Tune + Markov Heatmap)")

# Sidebar
st.sidebar.header("Controls & Tuning")
pick_count = st.sidebar.number_input("Prediction size (numbers)", 1, 5, value=1)
deterministic = st.sidebar.checkbox("Deterministic (top-k)", value=st.session_state.best_params["deterministic"])
temperature = st.sidebar.slider("Sampling temperature (if stochastic)", 0.4, 2.0, float(st.session_state.best_params["temperature"]))
half_life_ui = st.sidebar.slider("Half-life (decay) override", 5, 200, int(st.session_state.best_params["half_life"]))
streak_boost_ui = st.sidebar.slider("Streak boost override", 0.0, 2.0, float(st.session_state.best_params["streak_boost"]))
auto_predict = st.sidebar.checkbox("Auto-predict when adding draws", value=True)
show_debug = st.sidebar.checkbox("Show debug info", value=False)
if st.sidebar.button("Force sync all CSVs to R2"):
    save_df_to_r2(st.session_state.last_draws, F_LAST)
    save_df_to_r2(st.session_state.predictions, F_PRED)
    save_df_to_r2(st.session_state.accuracy_log, F_ACC)
    st.sidebar.success("Synced CSVs to R2")

# Manual input form (dropdowns)
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
    st.success("Draw added and synced to R2")
    if auto_predict:
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

# Paste area and Auto-map
st.write("---")
st.subheader("Paste lines (auto-map) / Bulk CSV Upload")
pasted = st.text_area("Paste lines from coinryze.org (issue_id,timestamp,number,color,size) or CSV content", height=140)
if st.button("Parse & Add Pasted Lines"):
    lines = [l.strip() for l in pasted.splitlines() if l.strip() != ""]
    added = 0
    for L in lines:
        parsed = parse_line_to_row(L)
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
            # try simple auto-map
            def auto_map_and_normalize_df(df_in):
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
                norm["number"] = norm["number"].apply(lambda x: try_int(x) if pd.notna(x) and str(x).strip()!="" else None)
                norm = norm.dropna(subset=["number"])
                norm["number"] = norm["number"].astype(int).astype(str)
                # infer missing color/size
                for idx, row in norm.iterrows():
                    n = try_int(row["number"])
                    if not row.get("color"):
                        norm.at[idx,"color"] = infer_color(n)
                    if not row.get("size"):
                        norm.at[idx,"size"] = infer_size(n)
                return norm[DRAW_COLS]
            mapped = auto_map_and_normalize_df(df_new)
            if mapped is not None and not mapped.empty:
                st.session_state.last_draws = pd.concat([mapped, st.session_state.last_draws], ignore_index=True)
                total_added += mapped.shape[0]
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if total_added:
        save_df_to_r2(st.session_state.last_draws, F_LAST)
        st.success(f"Appended {total_added} rows and synced to R2")

# Historical draws
st.write("---")
st.subheader("Historical draws (latest 50)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No draws yet")
else:
    st.dataframe(st.session_state.last_draws.head(50))

# Predict now / Use tuned params
st.write("---")
st.subheader("Smart Predictions (use tuned params or Run Now)")
st.write("Current tuned params:")
st.json(st.session_state.best_params)

if st.button("Run Prediction Now (apply current params)"):
    bp = st.session_state.best_params
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                             w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                             half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                             deterministic=bp["deterministic"], temperature=bp["temperature"])
    preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
    rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method":"det" if bp["deterministic"] else "stoch"}
    st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
    save_df_to_r2(st.session_state.predictions, F_PRED)
    st.success(f"Prediction saved: {preds}")

# Show latest saved prediction (do not re-generate; show saved)
st.write("---")
st.subheader("Latest saved prediction")
if st.session_state.predictions.shape[0] == 0:
    st.info("No saved predictions yet")
else:
    latest = st.session_state.predictions.iloc[0]
    try:
        latest_pred = json.loads(latest["prediction"])
    except Exception:
        latest_pred = []
    st.write("Saved at:", latest.get("created_at"))
    st.table(pd.DataFrame(latest_pred))

# Evaluate last saved prediction vs latest actual
st.write("---")
st.subheader("Evaluate last saved prediction vs latest actual draw")
if st.session_state.predictions.shape[0] == 0 or st.session_state.last_draws.shape[0] == 0:
    st.info("Need at least one saved prediction and one actual draw to evaluate.")
else:
    saved_pred = json.loads(st.session_state.predictions.iloc[0]["prediction"])
    latest_actual_row = st.session_state.last_draws.iloc[0]
    actual_num = try_int(latest_actual_row.get("number", None))
    st.write("Latest actual draw:", latest_actual_row.to_dict())
    st.write("Saved prediction:", saved_pred)
    if actual_num is None:
        st.warning("Latest actual draw number invalid.")
    else:
        saved_nums = [int(x["number"]) for x in saved_pred if "number" in x]
        overlap_pct = (len(set(saved_nums).intersection({actual_num})) / max(1, len(saved_nums))) * 100.0
        st.metric("Overlap % (numbers)", f"{overlap_pct:.1f}%")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(saved_pred), "actual": json.dumps([actual_num]), "accuracy_pct": overlap_pct}
            st.session_state.accuracy_log = pd.concat([pd.DataFrame([rec]), st.session_state.accuracy_log], ignore_index=True)
            save_df_to_r2(st.session_state.accuracy_log, F_ACC)
            st.success("Accuracy logged and synced")

# Backtest / AutoTune UI
st.write("---")
st.subheader("Backtest & Auto-Tune (Grid + Random Search)")
n_backtest = st.sidebar.number_input("Backtest window (last N draws)", min_value=20, max_value=2000, value=200, step=10)
grid_button = st.button("Run Grid backtest (deterministic, quick)")
rand_button = st.button("Run Randomized Auto-Tune (Bayesian-like, slower)")

if grid_button:
    with st.spinner("Running grid backtest..."):
        grid = build_grid_for_search()
        best, df_res = backtest_grid(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count, n_backtest=n_backtest, grid=grid)
        if best is None:
            st.warning("Not enough history to run backtest (need at least ~12 draws).")
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
            st.success("Grid backtest complete. Best params applied to session.")
            st.dataframe(df_res.head(10))
            st.download_button("Download grid results", data=df_res.to_csv(index=False).encode("utf-8"), file_name="grid_results.csv")

if rand_button:
    with st.spinner("Running randomized Auto-Tune (may take time)..."):
        best_rand, df_rand = randomized_search(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count, n_iter=60, n_backtest=n_backtest)
        if best_rand is None:
            st.warning("Auto-tune failed (not enough history).")
        else:
            st.session_state.best_params.update({
                "w_markov": float(best_rand["w_markov"]),
                "w_freq": float(best_rand["w_freq"]),
                "w_recency": float(best_rand["w_recency"]),
                "half_life": int(best_rand["half_life"]),
                "streak_boost": float(best_rand["streak_boost"]),
                "temperature": float(best_rand.get("temperature",1.0)),
                "deterministic": True
            })
            st.success("Randomized Auto-Tune complete. Best params applied to session.")
            st.dataframe(df_rand.head(10))
            st.download_button("Download randomized search results", data=df_rand.to_csv(index=False).encode("utf-8"), file_name="random_search_results.csv")

# Show ensemble prediction using applied best_params (live display)
st.write("---")
st.subheader("Current ensemble prediction (using applied tuned params)")
bp = st.session_state.best_params.copy()
# allow manual overrides from sidebar sliders
bp["half_life"] = half_life_ui
bp["streak_boost"] = streak_boost_ui
bp["deterministic"] = deterministic
bp["temperature"] = temperature

if st.button("Show current ensemble probabilities & prediction"):
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True), pick_count=pick_count,
                             w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                             half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                             deterministic=bp["deterministic"], temperature=bp["temperature"])
    preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
    st.write("Predicted block:")
    st.table(pd.DataFrame(preds))
    # also show Markov heatmap
    mat_counts, mat_probs = markov_transition_matrix(st.session_state.last_draws)
    fig = plot_markov_heatmap(mat_probs)
    st.pyplot(fig)

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

# R2 files list
st.write("---")
st.subheader("R2 files in bucket")
try:
    keys = r2_list_keys()
    if keys:
        for k in keys:
            st.write("-", k)
    else:
        st.info("No files or cannot list (check creds/permissions)")
except Exception as e:
    st.error(f"Could not list R2 keys: {e}")

# Debug info
if show_debug:
    st.write("DEBUG")
    st.write("last_draws rows:", st.session_state.last_draws.shape[0])
    st.write("predictions rows:", st.session_state.predictions.shape[0])
    st.write("accuracy rows:", st.session_state.accuracy_log.shape[0])
    st.json(st.session_state.best_params)

# Footer notes
st.markdown("""
**Tips**
- Use Grid backtest first (fast) to get a reasonable parameter set, then run Randomized Auto-Tune for finer tuning.
- Backtest optimizes a combined score (number overlap, color accuracy, size accuracy). You can change scoring weights in `score_prediction_vs_actual`.
- If predictions remain unchanged after adding new draws, run Randomized Auto-Tune / Grid again to re-fit weights to new history.
""")
