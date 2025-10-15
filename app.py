# app.py
"""
Coinryze Analyzer — Ensemble + Backtest/AutoTune
Copy-paste this file and run: streamlit run app.py

Requirements:
pip install streamlit pandas numpy boto3
(Optional for ML: scikit-learn — not required here)

Secrets:
Create .streamlit/secrets.toml with the R2 credentials (recommended).
Example (local .streamlit/secrets.toml):
[r2]
R2_KEY_ID = "7423969d6d623afd9ae23258a6cd2839"
R2_SECRET = "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c"
R2_BUCKET = "coinryze-analyzer"
R2_ENDPOINT = "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime, json, math, time

# ---------------------------
# R2 credentials (from secrets or fallback)
# ---------------------------
def load_r2_creds():
    try:
        r = st.secrets["r2"]
        return {
            "key": r.get("R2_KEY_ID") or r.get("key") or r.get("R2_KEY"),
            "secret": r.get("R2_SECRET") or r.get("secret") or r.get("R2_SECRET_KEY"),
            "bucket": r.get("R2_BUCKET") or r.get("bucket"),
            "endpoint": r.get("R2_ENDPOINT") or r.get("endpoint"),
        }
    except Exception:
        # fallback (convenience) — move these into secrets for production
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

# ---------------------------
# R2 helpers (safe reading/writing with column type normalization)
# ---------------------------
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

def read_csv_safe_from_bytes(b, cols=None):
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
        return read_csv_safe_from_bytes(b, cols=cols)
    if local_path:
        try:
            return pd.read_csv(local_path, dtype=str)
        except Exception:
            return pd.DataFrame(columns=cols if cols else [])
    return pd.DataFrame(columns=cols if cols else [])

def save_df_to_r2(df, key):
    # ensure pick_count int type before saving
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

# ---------------------------
# Session DataFrames initialization
# ---------------------------
if "last_draws" not in st.session_state:
    st.session_state.last_draws = load_df(F_LAST, cols=DRAW_COLS)
    # ensure canonical columns
    for c in DRAW_COLS:
        if c not in st.session_state.last_draws.columns:
            st.session_state.last_draws[c] = ""

if "predictions" not in st.session_state:
    st.session_state.predictions = load_df(F_PRED, cols=["created_at", "prediction", "pick_count", "algo", "method"])
    for c in ["created_at", "prediction", "pick_count", "algo", "method"]:
        if c not in st.session_state.predictions.columns:
            st.session_state.predictions[c] = ""
    # ensure pick_count integer
    if "pick_count" in st.session_state.predictions.columns:
        try:
            st.session_state.predictions["pick_count"] = pd.to_numeric(st.session_state.predictions["pick_count"], errors="coerce").fillna(1).astype(int)
        except Exception:
            st.session_state.predictions["pick_count"] = 1

if "accuracy_log" not in st.session_state:
    st.session_state.accuracy_log = load_df(F_ACC, cols=["timestamp", "predicted", "actual", "accuracy_pct"])
    for c in ["timestamp", "predicted", "actual", "accuracy_pct"]:
        if c not in st.session_state.accuracy_log.columns:
            st.session_state.accuracy_log[c] = ""

# defaults for tuned params (will be updated after backtest)
if "best_params" not in st.session_state:
    st.session_state.best_params = {
        "w_markov": 0.6,
        "w_freq": 0.25,
        "w_recency": 0.15,
        "half_life": 30,
        "streak_boost": 0.5,
        "temperature": 1.0,
        "deterministic": True
    }

# ---------------------------
# Helpers: parsing & inference
# ---------------------------
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

# ---------------------------
# Prediction engine (ensemble with stochastic option)
# ---------------------------
def build_markov(nums):
    K = 10
    mat = np.ones((K, K), dtype=float)  # add-one smoothing
    for a, b in zip(nums[:-1], nums[1:]):
        mat[a, b] += 1.0
    row_sums = mat.sum(axis=1, keepdims=True)
    return mat / row_sums

def freq_with_decay(nums, half_life):
    # exponential decay: recent draws weighted more
    weights = {}
    N = len(nums)
    for idx, v in enumerate(nums):
        age = N - 1 - idx  # 0 means most recent
        w = math.exp(-math.log(2) * (age / (half_life if half_life>0 else 1)))
        weights[v] = weights.get(v, 0.0) + w
    total = sum(weights.values()) if weights else 1.0
    return {i: weights.get(i, 0.0)/total for i in range(10)}

def compute_conditional_color_size(df):
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
    # smoothing and normalize
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
    # return last-run length for the most recent number
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
    """
    df_history expects most recent at end (oldest->newest).
    Returns list of integers (predicted numbers).
    """
    if df_history is None or df_history.shape[0] == 0:
        return []
    nums = [try_int(x) for x in df_history["number"].tolist() if try_int(x) is not None]
    if not nums:
        return []
    # Markov
    if len(nums) >= 2:
        markov_probs = build_markov(nums)
        last = nums[-1]
        markov_dist = markov_probs[last]  # numpy array length 10
    else:
        markov_dist = np.ones(10) / 10.0
    # Frequency with decay
    freq_dec = freq_with_decay(nums, half_life)
    freq_vec = np.array([freq_dec.get(i, 0.0) for i in range(10)])
    # Recency vector: inverse age normalized
    N = len(nums)
    last_indexes = {}
    for idx, v in enumerate(nums):
        last_indexes[v] = idx
    recency = np.zeros(10)
    for i in range(10):
        if i in last_indexes:
            age = N - 1 - last_indexes[i]
            recency[i] = 1.0/(1.0 + age)
    recency_vec = recency / (recency.sum() if recency.sum()>0 else 1.0)
    # Combine (normalize weights)
    w_total = w_markov + w_freq + w_recency
    if w_total <= 0:
        w_markov, w_freq, w_recency = 1.0, 0.0, 0.0
        w_total = 1.0
    w_markov_n = w_markov / w_total
    w_freq_n = w_freq / w_total
    w_recency_n = w_recency / w_total
    combined = w_markov_n * np.array(markov_dist) + w_freq_n * freq_vec + w_recency_n * recency_vec
    # Streak boost
    streaks = compute_streaks(nums)
    for n, length in streaks.items():
        if length > 1 and streak_boost > 0:
            combined[n] += streak_boost * (length - 1)
    # normalize
    combined = np.clip(combined, 0.0, None)
    if combined.sum() <= 0:
        combined = np.ones_like(combined) / len(combined)
    else:
        combined = combined / combined.sum()
    # temperature scaling for stochasticity
    if temperature != 1.0 and temperature > 0:
        logits = np.log(np.clip(combined, 1e-12, None)) / temperature
        exps = np.exp(logits - np.max(logits))
        combined = exps / exps.sum()
    # selection
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
    color_cond, size_cond = compute_conditional_color_size(df_history)
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

# ---------------------------
# Backtest / auto-tune (grid search)
# ---------------------------
def backtest_and_grid_search(df_history, n_backtest=200, pick_count=1, grid=None):
    """
    df_history: complete history with most recent at end (oldest->newest).
    n_backtest: use last n_backtest draws (simulate sequential predictions).
    grid: list of parameter dicts to evaluate.
    Returns: best_params, results_df
    """
    if df_history is None or df_history.shape[0] < 10:
        return None, pd.DataFrame()
    # make a clean numeric-only sequence
    seq_df = df_history.copy().reset_index(drop=True)
    seq_df = seq_df.dropna(subset=["number"])
    seq_df["number"] = seq_df["number"].astype(int)
    if seq_df.shape[0] < 10:
        return None, pd.DataFrame()
    total_len = seq_df.shape[0]
    # pick evaluation indices: we will simulate predicting the number at index t using history[:t]
    start_idx = max(5, total_len - n_backtest)
    indices = list(range(start_idx, total_len))  # predict for each of these
    results = []
    # iterate grid
    for params in grid:
        score_accum = 0.0
        count = 0
        for t in indices:
            hist = seq_df.iloc[:t]  # history up to before t
            actual = int(seq_df.iloc[t]["number"])
            # predict numbers
            nums = ensemble_predict(hist, pick_count=pick_count,
                                    w_markov=params["w_markov"],
                                    w_freq=params["w_freq"],
                                    w_recency=params["w_recency"],
                                    half_life=params["half_life"],
                                    streak_boost=params["streak_boost"],
                                    deterministic=True,
                                    temperature=params.get("temperature",1.0))
            # measure overlap (1 if actual in prediction, 0 otherwise) or fraction overlap
            overlap = 1.0 if actual in nums else 0.0
            score_accum += overlap
            count += 1
        avg_score = score_accum / count if count>0 else 0.0
        res = params.copy()
        res["score"] = avg_score
        results.append(res)
    results_df = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    best = results_df.iloc[0].to_dict() if not results_df.empty else None
    return best, results_df

# Build a reasonable default grid (keeps grid size manageable)
def build_grid():
    w_markov_vals = [0.2, 0.5, 0.8]
    w_freq_vals = [0.1, 0.3, 0.5]
    half_life_vals = [10, 30, 60]
    streak_vals = [0.0, 0.5, 1.0]
    temperature_vals = [0.6, 1.0, 1.4]
    grid = []
    for wm in w_markov_vals:
        for wf in w_freq_vals:
            # make sure there's some recency left
            wr_candidates = [0.0, 0.1, 0.2]  # small recency values to keep sum acceptable
            for wr in wr_candidates:
                for hl in half_life_vals:
                    for sb in streak_vals:
                        for temp in temperature_vals:
                            grid.append({
                                "w_markov": wm,
                                "w_freq": wf,
                                "w_recency": wr,
                                "half_life": hl,
                                "streak_boost": sb,
                                "temperature": temp
                            })
    return grid

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Coinryze Analyzer — AutoTune", layout="wide")
st.title("Coinryze Analyzer — Ensemble + Backtest AutoTune")

# Sidebar controls
st.sidebar.header("Controls")
pick_count = st.sidebar.number_input("Prediction size (pick_count)", min_value=1, max_value=5, value=1)
deterministic = st.sidebar.checkbox("Deterministic predictions (top-k)", value=True)
temperature = st.sidebar.slider("Sampling temperature (used when stochastic)", 0.4, 2.0, float(st.session_state.best_params["temperature"]))
half_life_default = st.sidebar.slider("Half-life (decay) default", 5, 200, int(st.session_state.best_params["half_life"]))
streak_default = st.sidebar.slider("Streak boost default", 0.0, 2.0, float(st.session_state.best_params["streak_boost"]))
auto_predict_toggle = st.sidebar.checkbox("Auto-predict when adding draws", value=True)
n_backtest = st.sidebar.number_input("Backtest window (last N draws to evaluate)", min_value=20, max_value=2000, value=200, step=10)
run_backtest = st.sidebar.button("Run Backtest / Auto-Tune (Grid Search)")

# Show current best params
st.sidebar.markdown("**Current applied params:**")
st.sidebar.json(st.session_state.best_params)

# Top: manual input form (dropdowns)
st.header("Manual Input — Add Single Draw")
with st.form("manual_form", clear_on_submit=True):
    c1,c2,c3,c4,c5 = st.columns([1.4,2.2,1,1,1])
    with c1:
        issue_id = st.text_input("issue_id")
    with c2:
        timestamp = st.text_input("timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
    with c3:
        number = st.selectbox("number (0-9)", list(range(10)))
    with c4:
        color = st.selectbox("color", ["Red","Green","Red-purple","Green-purple"])
    with c5:
        size = st.selectbox("size", ["Small","Big"])
    add = st.form_submit_button("Add Draw")

if add:
    row = {"issue_id": issue_id or datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
           "timestamp": timestamp,
           "number": str(number),
           "color": color,
           "size": size}
    st.session_state.last_draws = pd.concat([pd.DataFrame([row]), st.session_state.last_draws], ignore_index=True)
    save_df_to_r2(st.session_state.last_draws, F_LAST)
    st.success("Added draw and synced to R2")
    # Auto-predict using current best params
    if auto_predict_toggle:
        bp = st.session_state.best_params
        nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True),
                                 pick_count=pick_count,
                                 w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                                 half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                                 deterministic=bp["deterministic"], temperature=bp["temperature"])
        preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method": "det" if bp["deterministic"] else "stoch"}
        st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
        save_df_to_r2(st.session_state.predictions, F_PRED)
        st.info(f"Auto-prediction saved: {preds}")

# Paste/automap area
st.write("---")
st.subheader("Paste lines or CSV (Auto-map)")
pasted_text = st.text_area("Paste rows or CSV text (issue_id,timestamp,number,color,size) — header optional", height=160)
if st.button("Auto-map & Add pasted lines"):
    lines = [l.strip() for l in pasted_text.splitlines() if l.strip()]
    added = 0
    # try header CSV first
    try:
        if len(lines) > 1 and "," in lines[0] and ("issue" in lines[0].lower() or "number" in lines[0].lower()):
            df_try = pd.read_csv(StringIO("\n".join(lines)), dtype=str)
            mapped = auto_map_and_normalize_df(df_try) if 'auto_map_and_normalize_df' in globals() else None
            if mapped is not None and not mapped.empty:
                st.session_state.last_draws = pd.concat([mapped, st.session_state.last_draws], ignore_index=True)
                added += mapped.shape[0]
        # fallback line-by-line
    except Exception:
        pass
    for L in lines:
        parsed = parse_line(L)
        if parsed:
            st.session_state.last_draws = pd.concat([pd.DataFrame([parsed]), st.session_state.last_draws], ignore_index=True)
            added += 1
    if added:
        save_df_to_r2(st.session_state.last_draws, F_LAST)
        st.success(f"Added {added} rows and synced to R2")
        if auto_predict_toggle:
            bp = st.session_state.best_params
            nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True),
                                     pick_count=pick_count,
                                     w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                                     half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                                     deterministic=bp["deterministic"], temperature=bp["temperature"])
            preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method": "det" if bp["deterministic"] else "stoch"}
            st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
            save_df_to_r2(st.session_state.predictions, F_PRED)
            st.info(f"Auto-prediction saved: {preds}")
    else:
        st.warning("No valid pasted rows parsed. Check format.")

# Bulk uploader
st.write("---")
st.subheader("Bulk CSV Upload (Auto-map columns)")
uploaded = st.file_uploader("Upload CSV(s) with draws", accept_multiple_files=True, type=["csv"])
if uploaded:
    tot = 0
    for f in uploaded:
        try:
            df_new = pd.read_csv(f, dtype=str)
            # try to normalize
            mapped = auto_map_and_normalize_df(df_new) if 'auto_map_and_normalize_df' in globals() else None
            if mapped is None or mapped.empty:
                st.warning(f"{f.name} not auto-mapped; skipping")
                continue
            st.session_state.last_draws = pd.concat([mapped, st.session_state.last_draws], ignore_index=True)
            tot += mapped.shape[0]
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if tot:
        save_df_to_r2(st.session_state.last_draws, F_LAST)
        st.success(f"Appended {tot} rows and synced to R2")
        if auto_predict_toggle:
            bp = st.session_state.best_params
            nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True),
                                     pick_count=pick_count,
                                     w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                                     half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                                     deterministic=bp["deterministic"], temperature=bp["temperature"])
            preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method": "det" if bp["deterministic"] else "stoch"}
            st.session_state.predictions = pd.concat([pd.DataFrame([rec]), st.session_state.predictions], ignore_index=True)
            save_df_to_r2(st.session_state.predictions, F_PRED)
            st.info(f"Auto-prediction saved: {preds}")

# Display historical draws
st.write("---")
st.subheader("Historical draws (latest 50)")
if st.session_state.last_draws.shape[0] == 0:
    st.info("No draws yet.")
else:
    st.dataframe(st.session_state.last_draws.head(50))
    st.download_button("Download last_draws.csv", data=st.session_state.last_draws.to_csv(index=False).encode("utf-8"), file_name="last_draws.csv")

# Smart predictions (apply best_params or Run Now)
st.write("---")
st.subheader("Smart Predictions (use tuned params or run now)")
st.write("Current tuned params (editable in sidebar after backtest):")
st.json(st.session_state.best_params)

if st.button("Run Prediction Now (use current params)"):
    bp = st.session_state.best_params
    nums = ensemble_predict(st.session_state.last_draws[::-1].reset_index(drop=True),
                             pick_count=pick_count,
                             w_markov=bp["w_markov"], w_freq=bp["w_freq"], w_recency=bp["w_recency"],
                             half_life=bp["half_life"], streak_boost=bp["streak_boost"],
                             deterministic=bp["deterministic"], temperature=bp["temperature"])
    preds = predict_color_size_for_numbers(nums, st.session_state.last_draws, deterministic=bp["deterministic"])
    st.session_state.predictions = pd.concat([pd.DataFrame([{"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(preds), "pick_count": pick_count, "algo":"Ensemble", "method": "det" if bp["deterministic"] else "stoch"}]), st.session_state.predictions], ignore_index=True)
    save_df_to_r2(st.session_state.predictions, F_PRED)
    st.success(f"Prediction saved: {preds}")

# Show latest saved prediction
st.write("---")
st.subheader("Latest saved prediction")
if st.session_state.predictions.shape[0] == 0:
    st.info("No predictions saved.")
else:
    lp = st.session_state.predictions.iloc[0]
    try:
        lp_pred = json.loads(lp["prediction"])
    except Exception:
        lp_pred = []
    st.write("Saved at:", lp.get("created_at"))
    st.table(pd.DataFrame(lp_pred))

# Evaluate last saved prediction vs actual
st.write("---")
st.subheader("Evaluate last saved prediction vs latest actual draw")
if st.session_state.predictions.shape[0] == 0 or st.session_state.last_draws.shape[0] == 0:
    st.info("Need saved prediction and actual draw to evaluate.")
else:
    saved_pred = json.loads(st.session_state.predictions.iloc[0]["prediction"])
    latest = st.session_state.last_draws.iloc[0]
    actual_num = try_int(latest.get("number", None))
    st.write("Latest actual draw:", latest.to_dict())
    st.write("Saved prediction:", saved_pred)
    if actual_num is None:
        st.warning("Latest actual number invalid.")
    else:
        saved_nums = [int(x["number"]) for x in saved_pred if "number" in x]
        overlap = len(set(saved_nums).intersection({actual_num})) / max(1, len(saved_nums)) * 100.0
        st.metric("Overlap %", f"{overlap:.1f}%")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(saved_pred), "actual": json.dumps([actual_num]), "accuracy_pct": overlap}
            st.session_state.accuracy_log = pd.concat([pd.DataFrame([rec]), st.session_state.accuracy_log], ignore_index=True)
            save_df_to_r2(st.session_state.accuracy_log, F_ACC)
            st.success("Accuracy logged and synced.")

# Backtest / AutoTune
st.write("---")
st.subheader("Backtest & Auto-Tune (Grid Search)")
st.markdown("This will simulate predicting each draw in the last N draws using only prior history, test a grid of parameter combinations, and select the best parameters (maximize overlap).")

if run_backtest:
    st.info("Starting grid search — this may take time for large N. Progress bar will show.")
    grid = build_grid()
    best, results_df = backtest_and_grid_search(st.session_state.last_draws[::-1].reset_index(drop=True), n_backtest=n_backtest, pick_count=pick_count, grid=grid)
    if best is None:
        st.warning("Not enough history to run backtest (need at least 10 valid draws).")
    else:
        st.success("Grid search completed.")
        st.write("Top parameter combinations (sample):")
        st.dataframe(results_df.head(10))
        st.write("Best params found:")
        st.json(best)
        # apply best
        st.session_state.best_params.update({
            "w_markov": float(best["w_markov"]),
            "w_freq": float(best["w_freq"]),
            "w_recency": float(best["w_recency"]),
            "half_life": int(best["half_life"]),
            "streak_boost": float(best["streak_boost"]),
            "temperature": float(best.get("temperature", 1.0)),
            "deterministic": True
        })
        st.success("Applied best params to session. Run Prediction Now to use them.")
        # allow export results
        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download grid search results (CSV)", data=csv_bytes, file_name="grid_results.csv")

# Accuracy log & chart
st.write("---")
st.subheader("Accuracy log")
if st.session_state.accuracy_log.shape[0] == 0:
    st.info("No accuracy logs yet.")
else:
    df_acc = st.session_state.accuracy_log.copy()
    df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"], errors="coerce")
    df_acc["accuracy_pct"] = pd.to_numeric(df_acc["accuracy_pct"], errors="coerce").fillna(0)
    st.line_chart(df_acc.set_index("timestamp")["accuracy_pct"])
    st.dataframe(df_acc.head(200))
    st.download_button("Download accuracy_log.csv", data=df_acc.to_csv(index=False).encode("utf-8"), file_name="accuracy_log.csv")

# R2 bucket files list & downloads
st.write("---")
st.subheader("R2 bucket files")
try:
    keys = r2_list_keys()
    if keys:
        for k in keys:
            st.write("•", k)
            b = r2_get_bytes(k)
            if b is not None:
                st.download_button(f"Download {k}", data=b, file_name=k)
    else:
        st.info("No files found in R2 bucket.")
except Exception as e:
    st.error(f"Unable to list R2 keys: {e}")

# Debug
if st.sidebar.checkbox("Show debug info"):
    st.write("DEBUG INFO")
    st.write("Rows in last_draws:", st.session_state.last_draws.shape)
    st.write("Rows in predictions:", st.session_state.predictions.shape)
    st.write("Rows in accuracy_log:", st.session_state.accuracy_log.shape)
    st.write("Current best_params:", st.session_state.best_params)

st.markdown("""
**How to use (quick)**

1. Place your R2 credentials into `.streamlit/secrets.toml` as shown in the header (recommended for Render).
2. Add draws manually (dropdowns), paste lines, or bulk upload CSVs.
3. Use **Run Backtest / Auto-Tune** to search for the best ensemble weights using your history (choose N draws).
4. Click **Run Prediction Now** to create a new prediction using the tuned params.
5. Evaluate and Log accuracy, then iterate.

If you want, I can:
- Make backtest use color+size scoring (currently optimizes number overlap).
- Add an automatic tuner that periodically retrains and stores tuned params.
- Add faster randomized grid search (Bayesian) for larger datasets.
""")
