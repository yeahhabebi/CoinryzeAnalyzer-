# app.py
"""
Coinryze Analyzer - Advanced single-file Streamlit app with Cloudflare R2 sync
Features:
- Manual single draw input (issue_id,timestamp,number,color,size)
- Paste lines (coinryze.org) and Auto-map pasted CSVs with different column names
- Bulk CSV upload
- Color/Size inference rule options
- Prediction algorithms: Frequency+Recency, Markov chain, ML (RandomForest)
- Train ML model, One-click Predict, Save predictions, Log accuracy
- Sync CSVs to/from Cloudflare R2
- Uses st.secrets["r2"] if available; instructions provided below to set on Render.

How to set R2 credentials in Streamlit secrets (recommended for Render):
Create a file `secrets.toml` locally for testing (~/.streamlit/secrets.toml) or set Render environment secrets:
[ r2 ]
key = "YOUR_R2_ACCESS_KEY_ID"
secret = "YOUR_R2_SECRET_ACCESS_KEY"
bucket = "coinryze-analyzer"
endpoint = "https://<account_id>.r2.cloudflarestorage.com"
Then in the app access via: st.secrets["r2"]["key"], etc.

Requirements (add to requirements.txt):
streamlit
boto3
pandas
numpy
scikit-learn     # optional, for ML model (if missing, ML will be disabled)
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

# Optional ML dependencies
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --------------------------
# Configuration / R2 Credentials
# --------------------------
# Default (in-file) fallback credentials (already provided earlier).
# For security, move them to Streamlit secrets (recommended).
DEFAULT_R2 = {
    "key": "7423969d6d623afd9ae23258a6cd2839",
    "secret": "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c",
    "bucket": "coinryze-analyzer",
    "endpoint": "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"
}

# Use st.secrets["r2"] if present (recommended)
if "r2" in st.secrets:
    R2 = {
        "key": st.secrets["r2"].get("key", DEFAULT_R2["key"]),
        "secret": st.secrets["r2"].get("secret", DEFAULT_R2["secret"]),
        "bucket": st.secrets["r2"].get("bucket", DEFAULT_R2["bucket"]),
        "endpoint": st.secrets["r2"].get("endpoint", DEFAULT_R2["endpoint"]),
    }
else:
    R2 = DEFAULT_R2.copy()

R2_KEY_ID = R2["key"]
R2_SECRET = R2["secret"]
R2_BUCKET = R2["bucket"]
R2_ENDPOINT = R2["endpoint"]

# Filenames in bucket
F_LAST_DRAWS = "last_draws.csv"
F_PREDICTIONS = "predictions.csv"
F_ACCURACY = "accuracy_log.csv"

# canonical draw columns
DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

# --------------------------
# R2 client helpers
# --------------------------
def r2_client():
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_KEY_ID,
        aws_secret_access_key=R2_SECRET,
    )

def r2_get_bytes(key):
    s3 = r2_client()
    try:
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except ClientError:
        return None
    except Exception:
        return None

def r2_put_bytes(key, bts, content_type="text/csv"):
    s3 = r2_client()
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=bts, ContentType=content_type)
        return True
    except Exception as e:
        st.error(f"R2 upload error for {key}: {e}")
        return False

def r2_list_keys(prefix=""):
    s3 = r2_client()
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except Exception:
        return []

# --------------------------
# Data load/save utilities
# --------------------------
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

def save_df_to_r2(df, key):
    if df is None:
        df = pd.DataFrame()
    b = df.to_csv(index=False).encode("utf-8")
    return r2_put_bytes(key, b)

# --------------------------
# Initialize dataframes (load from R2 if present)
# --------------------------
last_draws = load_df_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv", cols=DRAW_COLS)
predictions_df = load_df_from_r2_or_local(F_PREDICTIONS, cols=["created_at", "prediction", "pick_count", "algo"])
accuracy_log = load_df_from_r2_or_local(F_ACCURACY, cols=["timestamp", "predicted", "actual", "accuracy_pct"])

if last_draws is None:
    last_draws = pd.DataFrame(columns=DRAW_COLS)
if predictions_df is None:
    predictions_df = pd.DataFrame(columns=["created_at", "prediction", "pick_count", "algo"])
if accuracy_log is None:
    accuracy_log = pd.DataFrame(columns=["timestamp", "predicted", "actual", "accuracy_pct"])

# Ensure columns exist
for col in DRAW_COLS:
    if col not in last_draws.columns:
        last_draws[col] = ""

# --------------------------
# Mapping & inference rules
# --------------------------
def infer_size_from_number_ruleA(n):
    # rule A (previous): 0-4 -> Small, 5-9 -> Big
    try:
        n = int(n)
    except:
        return ""
    return "Small" if 0 <= n <= 4 else "Big"

def infer_color_from_number_ruleA(n):
    # rule A: 0->Red-purple, 5->Green-purple, else 0-4->Red, 5-9->Green
    try:
        n = int(n)
    except:
        return ""
    if n == 0:
        return "Red-purple"
    if n == 5:
        return "Green-purple"
    return "Red" if 0 <= n <= 4 else "Green"

def infer_size_from_number_ruleB(n):
    # rule B (alternate): even -> Small, odd -> Big
    try:
        n = int(n)
    except:
        return ""
    return "Small" if n % 2 == 0 else "Big"

def infer_color_from_number_ruleB(n):
    # rule B (alternate): 0-2 Red, 3-5 Red-purple, 6-7 Green-purple, 8-9 Green (example)
    try:
        n = int(n)
    except:
        return ""
    if n <= 2:
        return "Red"
    if 3 <= n <= 5:
        return "Red-purple"
    if 6 <= n <= 7:
        return "Green-purple"
    return "Green"

# Choose inference function dynamically later via sidebar selection

# --------------------------
# Prediction algorithms
# --------------------------
def predict_freq_recency(df_draws, pick_count=1):
    # frequency + recency penalty (original)
    if df_draws is None or df_draws.shape[0] == 0:
        return []
    nums = []
    for v in df_draws["number"].astype(str).values:
        try:
            nums.append(int(v))
        except:
            continue
    if not nums:
        return []
    freq = pd.Series(nums).value_counts().to_dict()
    # recency index: reversed history, 0 => most recent
    last_index = {}
    for idx, v in enumerate(df_draws["number"].astype(str).values[::-1]):
        try:
            val = int(v)
            if val not in last_index:
                last_index[val] = idx
        except:
            continue
    scores = {}
    for n in range(0, 10):
        f = freq.get(n, 0)
        rec = last_index.get(n, 10000)
        scores[n] = f * 1.0 - rec * 0.2
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picks = [n for n, s in sorted_nums[:pick_count]]
    return picks

def predict_markov(df_draws, pick_count=1):
    # Build transition matrix from sequential numbers and follow transitions from last observed number
    if df_draws is None or df_draws.shape[0] < 2:
        # fallback to frequency
        return predict_freq_recency(df_draws, pick_count)
    seq = []
    for v in df_draws["number"].astype(str).values:
        try:
            seq.append(int(v))
        except:
            continue
    if len(seq) < 2:
        return predict_freq_recency(df_draws, pick_count)
    # transition counts 10x10
    trans = np.zeros((10, 10), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        trans[a, b] += 1
    # normalize rows to probabilities
    row_sums = trans.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(trans, row_sums, where=row_sums != 0)
    last = seq[-1]
    picks = []
    # greedily pick next numbers by highest transition probability; if not available fall back to freq
    if row_sums[last] == 0:
        return predict_freq_recency(df_draws, pick_count)
    next_probs = probs[last]
    sorted_next = np.argsort(-next_probs)
    for n in sorted_next:
        if len(picks) >= pick_count:
            break
        if next_probs[n] > 0:
            picks.append(int(n))
    # If transition yields fewer than pick_count, fill with freq-based
    if len(picks) < pick_count:
        fill = predict_freq_recency(df_draws, pick_count=pick_count - len(picks))
        for f in fill:
            if f not in picks:
                picks.append(f)
    return picks[:pick_count]

# ML based predictor (RandomForest multiclass)
ML_MODEL_KEY = "ml_model_obj"
def prepare_ml_dataset(df_draws, window=3):
    """
    Prepare features: for each position t, features are numbers at t-window..t-1
    label is number at t
    """
    seq = []
    for v in df_draws["number"].astype(str).values:
        try:
            seq.append(int(v))
        except:
            continue
    if len(seq) <= window:
        return None, None
    X = []
    y = []
    for i in range(window, len(seq)):
        X.append(seq[i-window:i])
        y.append(seq[i])
    X = np.array(X)  # shape N x window
    y = np.array(y)
    # One-hot encode numeric features (0-9) per position -> vector length 10*window
    N, W = X.shape
    X_onehot = np.zeros((N, 10 * W), dtype=int)
    for i in range(N):
        for j in range(W):
            val = X[i, j]
            if 0 <= val <= 9:
                X_onehot[i, j * 10 + val] = 1
    return X_onehot, y

def train_ml_model(df_draws, window=3):
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn not available. Install scikit-learn to use ML predictor.")
        return None, None
    X, y = prepare_ml_dataset(df_draws, window=window)
    if X is None:
        st.warning("Not enough history to train ML model.")
        return None, None
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    # Evaluate quickly (train accuracy)
    y_pred = clf.predict(X)
    acc = float((y_pred == y).mean())
    return clf, acc

def predict_with_ml(clf, df_draws, window=3, pick_count=1):
    if clf is None:
        return []
    # get last window numbers
    seq = []
    for v in df_draws["number"].astype(str).values:
        try:
            seq.append(int(v))
        except:
            continue
    if len(seq) < window:
        return []
    features = np.zeros((1, 10 * window), dtype=int)
    last_window = seq[-window:]
    for j, val in enumerate(last_window):
        if 0 <= val <= 9:
            features[0, j * 10 + val] = 1
    # Predict probabilities for next number
    try:
        probs = clf.predict_proba(features)[0]  # classes 0..9 not guaranteed order; sklearn classes_ maps
        classes = clf.classes_
        # build (class, prob) pairs
        pairs = list(zip(classes, probs))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        picks = [int(c) for c, p in pairs_sorted[:pick_count]]
        return picks
    except Exception as e:
        # fallback to predict
        try:
            pred = clf.predict(features)[0]
            return [int(pred)]
        except Exception:
            return []

# --------------------------
# Parsing helpers (pasted lines) and Auto-map
# --------------------------
def parse_line_to_row(line):
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        return None
    # try to be permissive: allow missing color/size
    issue_id = parts[0] if len(parts) > 0 else ""
    timestamp = parts[1] if len(parts) > 1 else ""
    number = parts[2] if len(parts) > 2 else ""
    color = parts[3] if len(parts) > 3 else ""
    size = parts[4] if len(parts) > 4 else ""
    try:
        number_val = str(int(float(number)))
    except:
        number_val = ""
    return {
        "issue_id": issue_id,
        "timestamp": timestamp,
        "number": number_val,
        "color": color,
        "size": size,
    }

def auto_map_and_normalize_df(df_in):
    """
    Try to remap columns to canonical DRAW_COLS using heuristics.
    Returns normalized DataFrame with columns DRAW_COLS.
    """
    colmap = {}
    cols = list(df_in.columns)
    lower_cols = [c.lower() for c in cols]
    for canonical in DRAW_COLS:
        # find best match
        if canonical in lower_cols:
            colmap[cols[lower_cols.index(canonical)]] = canonical
            continue
        # heuristics
        if canonical == "issue_id":
            candidates = [c for c in cols if "issue" in c.lower() or "id" in c.lower()]
        elif canonical == "timestamp":
            candidates = [c for c in cols if "time" in c.lower() or "date" in c.lower()]
        elif canonical == "number":
            candidates = [c for c in cols if "num" in c.lower() or c.lower() in ["n","number","value"]]
        elif canonical == "color":
            candidates = [c for c in cols if "color" in c.lower() or "colour" in c.lower()]
        elif canonical == "size":
            candidates = [c for c in cols if "size" in c.lower() or "big" in c.lower() or "small" in c.lower()]
        else:
            candidates = []
        if candidates:
            colmap[candidates[0]] = canonical
    # Build normalized DF
    norm = pd.DataFrame(columns=DRAW_COLS)
    for orig_col, canon in colmap.items():
        norm[canon] = df_in[orig_col].astype(str)
    # fill missing columns
    for c in DRAW_COLS:
        if c not in norm.columns:
            norm[c] = ""
    # ensure number column is normalized to single-digit string
    norm["number"] = norm["number"].apply(lambda x: try_cast_number(x))
    return norm[DRAW_COLS]

def try_cast_number(x):
    try:
        return str(int(float(str(x).strip())))
    except:
        return ""

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Coinryze Analyzer - Advanced", layout="wide")
st.title("Coinryze Analyzer — Advanced Predictions & R2 Sync")

# Sidebar controls
st.sidebar.header("Settings")
pick_count = st.sidebar.number_input("Prediction size (how many numbers)", min_value=1, max_value=5, value=1, step=1)
inference_rule = st.sidebar.selectbox("Color/Size inference rule", ("Rule A (default)", "Rule B (alternate)"))
algo_choice = st.sidebar.selectbox("Prediction algorithm", ("Frequency+Recency", "Markov", "ML (RandomForest)"))
ml_window = st.sidebar.slider("ML window (how many previous numbers used as features)", 1, 5, 3)
train_ml_btn = st.sidebar.button("Train ML model (RandomForest)")
show_debug = st.sidebar.checkbox("Show debug info", value=False)
auto_predict_toggle = st.sidebar.checkbox("Auto-predict after adding draws", value=True)
auto_map_btn = st.sidebar.button("Auto-map pasted CSV (if pasted)")

# pick inference functions
if inference_rule == "Rule A (default)":
    infer_color = infer_color_from_number_ruleA
    infer_size = infer_size_from_number_ruleA
else:
    infer_color = infer_color_from_number_ruleB
    infer_size = infer_size_from_number_ruleB

# Left column: Input and paste
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Manual Input — Add Single Last Draw")
    with st.form("single_draw", clear_on_submit=True):
        c1, c2, c3, c4, c5 = st.columns([1.2,1.8,1,1,1])
        with c1:
            issue_id = st.text_input("issue_id", placeholder="e.g. 202510131045")
        with c2:
            timestamp = st.text_input("timestamp", value=datetime.datetime.utcnow().strftime("%H:%M:%S %m/%d/%Y"))
        with c3:
            number = st.text_input("number (0-9)", placeholder="e.g. 3")
        with c4:
            color = st.text_input("color (optional)", value="", placeholder="e.g. Green")
        with c5:
            size = st.text_input("size (optional)", value="", placeholder="Small/Big")
        submitted_single = st.form_submit_button("Add draw row")
    if submitted_single:
        if not number:
            st.error("Please provide a number 0-9.")
        else:
            if color == "" or size == "":
                if color == "":
                    color = infer_color(number)
                if size == "":
                    size = infer_size(number)
            new_row = {
                "issue_id": str(issue_id) if issue_id else datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
                "timestamp": str(timestamp),
                "number": str(int(float(number))),
                "color": color,
                "size": size,
            }
            last_draws = pd.concat([pd.DataFrame([new_row]), last_draws], ignore_index=True)
            save_df_to_r2(last_draws, F_LAST_DRAWS)
            st.success("Draw added and synced to R2.")
            # auto predict if toggled
            if auto_predict_toggle:
                if algo_choice == "Frequency+Recency":
                    picks = predict_freq_recency(last_draws, pick_count=pick_count)
                elif algo_choice == "Markov":
                    picks = predict_markov(last_draws, pick_count=pick_count)
                else:
                    # ML
                    clf, acc = (None, None)
                    # try to train quickly with available history
                    if SKLEARN_AVAILABLE:
                        clf, _ = train_ml_model(last_draws, window=ml_window)
                    picks = predict_with_ml(clf, last_draws, window=ml_window, pick_count=pick_count) if clf is not None else predict_freq_recency(last_draws, pick_count=pick_count)
                rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count, "algo": algo_choice}
                predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
                save_df_to_r2(predictions_df, F_PREDICTIONS)
                st.info(f"Auto-prediction saved: {picks}")

    st.write("---")
    st.subheader("Paste lines from coinryze.org (one per line)")
    sample = "202510131045, 15:25:00 10/13/2025, 3, Green , Small"
    multiline = st.text_area("Paste CSV lines here (issue_id,timestamp,number,color,size)", height=160, placeholder=sample)
    if st.button("Parse & Add Pasted Lines"):
        lines = [l.strip() for l in multiline.splitlines() if l.strip() != ""]
        added = 0
        for L in lines:
            parsed = parse_line_to_row(L)
            if parsed:
                if parsed["color"] == "" or parsed["size"] == "":
                    parsed["color"] = parsed["color"] or infer_color(parsed["number"])
                    parsed["size"] = parsed["size"] or infer_size(parsed["number"])
                last_draws = pd.concat([pd.DataFrame([parsed]), last_draws], ignore_index=True)
                added += 1
        if added:
            save_df_to_r2(last_draws, F_LAST_DRAWS)
            st.success(f"Added {added} pasted rows and synced to R2.")
            if auto_predict_toggle:
                # auto predict using chosen algorithm
                if algo_choice == "Frequency+Recency":
                    picks = predict_freq_recency(last_draws, pick_count=pick_count)
                elif algo_choice == "Markov":
                    picks = predict_markov(last_draws, pick_count=pick_count)
                else:
                    clf, _ = (None, None)
                    if SKLEARN_AVAILABLE:
                        clf, _ = train_ml_model(last_draws, window=ml_window)
                    picks = predict_with_ml(clf, last_draws, window=ml_window, pick_count=pick_count) if clf is not None else predict_freq_recency(last_draws, pick_count=pick_count)
                rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count, "algo": algo_choice}
                predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
                save_df_to_r2(predictions_df, F_PREDICTIONS)
                st.info(f"Auto-prediction saved: {picks}")
        else:
            st.warning("No valid rows parsed. Check format (issue_id,timestamp,number,color,size).")

    if auto_map_btn and multiline.strip() != "":
        # attempt to parse pasted text as CSV and remap
        try:
            df_temp = pd.read_csv(StringIO(multiline), header=0, dtype=str)
            mapped = auto_map_and_normalize_df(df_temp)
            if not mapped.empty:
                last_draws = pd.concat([mapped, last_draws], ignore_index=True)
                save_df_to_r2(last_draws, F_LAST_DRAWS)
                st.success("Auto-mapped pasted CSV and synced to R2.")
            else:
                st.warning("Auto-mapping produced no rows. Inspect pasted CSV format.")
        except Exception as e:
            st.error(f"Auto-map failed: {e}")

    st.write("---")
    st.subheader("Bulk CSV Upload (one or more files)")
    uploaded_files = st.file_uploader("Upload CSV(s) with columns (try to include issue_id,timestamp,number)", accept_multiple_files=True, type=["csv"])
    if uploaded_files:
        total_added = 0
        for f in uploaded_files:
            try:
                df_new = pd.read_csv(f, dtype=str)
                # Auto map and normalize
                mapped = auto_map_and_normalize_df(df_new)
                # infer color/size where missing
                for idx, row in mapped.iterrows():
                    if row["color"] == "" or row["size"] == "":
                        mapped.at[idx, "color"] = row["color"] or infer_color(mapped.at[idx, "number"])
                        mapped.at[idx, "size"] = row["size"] or infer_size(mapped.at[idx, "number"])
                last_draws = pd.concat([mapped, last_draws], ignore_index=True)
                total_added += mapped.shape[0]
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
        if total_added:
            save_df_to_r2(last_draws, F_LAST_DRAWS)
            st.success(f"Appended {total_added} rows from uploaded files and synced to R2.")
            if auto_predict_toggle:
                if algo_choice == "Frequency+Recency":
                    picks = predict_freq_recency(last_draws, pick_count=pick_count)
                elif algo_choice == "Markov":
                    picks = predict_markov(last_draws, pick_count=pick_count)
                else:
                    clf, _ = (None, None)
                    if SKLEARN_AVAILABLE:
                        clf, _ = train_ml_model(last_draws, window=ml_window)
                    picks = predict_with_ml(clf, last_draws, window=ml_window, pick_count=pick_count) if clf is not None else predict_freq_recency(last_draws, pick_count=pick_count)
                rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count, "algo": algo_choice}
                predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
                save_df_to_r2(predictions_df, F_PREDICTIONS)
                st.info(f"Auto-prediction saved: {picks}")

with col2:
    st.subheader("Smart Predictions")
    st.markdown("Choose algorithm in sidebar. Click **Train ML model** to train RandomForest (if available).")
    # Train ML if requested
    if train_ml_btn:
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn is not installed. Add `scikit-learn` to requirements.txt to use ML.")
        else:
            with st.spinner("Training ML model..."):
                try:
                    clf, train_acc = train_ml_model(last_draws, window=ml_window)
                    if clf is not None:
                        st.success(f"ML model trained (in-sample accuracy {train_acc*100:.1f}%). You can now use ML predictions.")
                        st.session_state["ml_model"] = clf
                    else:
                        st.warning("Not enough history to train ML model.")
                except Exception as e:
                    st.error(f"ML training failed: {e}\n{traceback.format_exc()}")

    # show prediction for chosen algorithm
    if algo_choice == "Frequency+Recency":
        picks = predict_freq_recency(last_draws, pick_count=pick_count)
    elif algo_choice == "Markov":
        picks = predict_markov(last_draws, pick_count=pick_count)
    else:
        clf = st.session_state.get("ml_model", None) if "ml_model" in st.session_state else None
        if clf is None and SKLEARN_AVAILABLE:
            # try quick train (in-memory) for convenience
            clf, _ = train_ml_model(last_draws, window=ml_window)
            st.session_state["ml_model"] = clf
        picks = predict_with_ml(clf, last_draws, window=ml_window, pick_count=pick_count) if clf is not None else predict_freq_recency(last_draws, pick_count=pick_count)
    st.write("Predicted next numbers:", picks)
    # show color/size suggestions
    pick_details = []
    for p in picks:
        pick_details.append({"number": p, "color": infer_color(p), "size": infer_size(p)})
    if pick_details:
        st.table(pd.DataFrame(pick_details))
    if st.button("Save this prediction"):
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count, "algo": algo_choice}
        predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
        save_df_to_r2(predictions_df, F_PREDICTIONS)
        st.success("Prediction saved and synced to R2.")

# --------------------------
# Historical draws & Predictions table
# --------------------------
st.write("---")
st.subheader("Historical draws (top 50)")
if last_draws is None or last_draws.shape[0] == 0:
    st.info("No historical draws yet. Add via manual input, paste, or upload.")
else:
    st.dataframe(last_draws.head(50))

st.subheader("Predictions history (top 50)")
if predictions_df is None or predictions_df.shape[0] == 0:
    st.info("No predictions yet.")
else:
    st.dataframe(predictions_df.head(50))
    st.download_button("Download predictions.csv", data=predictions_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# --------------------------
# Evaluate / log accuracy
# --------------------------
st.write("---")
st.subheader("Evaluate last prediction vs latest actual draw (Overlap%)")
if predictions_df.shape[0] > 0 and last_draws.shape[0] > 0:
    try:
        last_pred = json.loads(predictions_df.iloc[0]["prediction"])
    except Exception:
        last_pred = []
    latest_row = last_draws.iloc[0]
    try:
        latest_num = int(latest_row["number"])
    except Exception:
        latest_num = None
    st.write("Last prediction:", last_pred)
    st.write("Latest actual:", latest_num)
    if latest_num is not None and last_pred:
        overlap_pct = (len(set(last_pred).intersection({latest_num})) / max(1, len(last_pred))) * 100.0
        st.write(f"Overlap with latest draw: **{overlap_pct:.1f}%**")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(last_pred), "actual": json.dumps([latest_num]), "accuracy_pct": overlap_pct}
            accuracy_log = pd.concat([pd.DataFrame([rec]), accuracy_log], ignore_index=True)
            save_df_to_r2(accuracy_log, F_ACCURACY)
            st.success("Logged accuracy to accuracy_log.csv and synced to R2.")
else:
    st.info("Need at least one saved prediction and one actual draw to evaluate.")

# --------------------------
# R2 Files and utilities
# --------------------------
st.write("---")
st.subheader("R2 Files & Utilities")
try:
    keys = r2_list_keys()
    if keys:
        st.write("Files in R2 bucket:")
        for k in keys:
            st.write("-", k)
            b = r2_get_bytes(k)
            if b is not None:
                st.download_button(f"Download {k}", data=BytesIO(b), file_name=k)
    else:
        st.info("No files found in R2 bucket.")
except Exception as e:
    st.error(f"Failed to list R2 keys: {e}")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Force sync all CSVs to R2"):
        save_df_to_r2(last_draws, F_LAST_DRAWS)
        save_df_to_r2(predictions_df, F_PREDICTIONS)
        save_df_to_r2(accuracy_log, F_ACCURACY)
        st.success("All CSVs uploaded to R2.")
with col_b:
    if st.button("Reload CSVs from R2"):
        ld = load_df_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv", cols=DRAW_COLS)
        pdx = load_df_from_r2_or_local(F_PREDICTIONS)
        agr = load_df_from_r2_or_local(F_ACCURACY)
        if ld is not None and ld.shape[0] > 0:
            last_draws = ld
        if pdx is not None:
            predictions_df = pdx
        if agr is not None:
            accuracy_log = agr
        st.success("Reloaded CSVs from R2 (if present). Please scroll to sections to view updated tables.")

# --------------------------
# Accuracy trend chart
# --------------------------
st.write("---")
st.subheader("Accuracy Tracker & Trend")
if accuracy_log is not None and accuracy_log.shape[0] > 0:
    try:
        accuracy_log["timestamp"] = pd.to_datetime(accuracy_log["timestamp"])
        df_acc = accuracy_log.sort_values("timestamp", ascending=True)
        st.line_chart(df_acc.set_index("timestamp")["accuracy_pct"])
    except Exception as e:
        st.error(f"Failed to render accuracy chart: {e}")
else:
    st.info("No accuracy logs yet.")

# --------------------------
# Debug info
# --------------------------
if show_debug:
    st.write("DEBUG")
    st.write("R2 endpoint:", R2_ENDPOINT)
    st.write("R2 bucket:", R2_BUCKET)
    st.write("Number of draws:", last_draws.shape[0])
    st.write("Predictions rows:", predictions_df.shape[0])
    st.write("Accuracy rows:", accuracy_log.shape[0])
    try:
        st.write("R2 keys:", r2_list_keys())
    except Exception as e:
        st.write("R2 keys list failed:", e)

# --------------------------
# Footer quick help
# --------------------------
st.markdown("""
## Quick notes
- To move R2 keys to Streamlit secrets (recommended on Render):
  - In Render dashboard, set a secret named `R2_KEY_ID`, `R2_SECRET`, `R2_BUCKET`, `R2_ENDPOINT` OR use `st.secrets` TOML with `[r2]` table as shown in the header comment.
  - Then replace DEFAULT_R2 or ensure `st.secrets['r2']` contains the keys.
- To enable ML model, add `scikit-learn` to your requirements.txt and redeploy.
- If you want a different/advanced ML pipeline, I can add feature engineering, hyperparameter tuning, and cross-validation.
""")
