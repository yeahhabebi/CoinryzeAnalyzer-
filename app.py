# app.py
"""
Coinryze Analyzer (final) - supports issue_id,timestamp,number,color,size format
Features:
- Manual single draw input (fields or paste CSV line)
- Paste multiple lines from coinryze.org format
- Bulk CSV upload
- Historical draws (top 50)
- Auto-predict toggle (predict next result(s) immediately after adding)
- R2 sync (last_draws.csv, predictions.csv, accuracy_log.csv)
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO, StringIO
import datetime
import json

# --------------------------
# R2 Configuration (already provided by you)
# --------------------------
R2_KEY_ID = "7423969d6d623afd9ae23258a6cd2839"
R2_SECRET = "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c"
R2_BUCKET = "coinryze-analyzer"
R2_ENDPOINT = "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"

# filenames
F_LAST_DRAWS = "last_draws.csv"
F_PREDICTIONS = "predictions.csv"
F_ACCURACY = "accuracy_log.csv"

# S3/R2 client
s3 = boto3.client(
    "s3",
    region_name="auto",
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET,
)

# --------------------------
# Helpers: R2 interactions
# --------------------------
def r2_get_bytes(key):
    try:
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp["Body"].read()
    except ClientError:
        return None

def r2_put_bytes(key, bts, content_type="text/csv"):
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=bts, ContentType=content_type)
        return True
    except ClientError as e:
        st.error(f"R2 upload error: {e}")
        return False

def r2_list_keys(prefix=""):
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o["Key"] for o in resp.get("Contents", [])]
    except ClientError:
        return []

# --------------------------
# Data loading / initialisation
# --------------------------
def load_df_from_r2_or_local(key, local_path=None, cols=None):
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
    # fallback empty
    if cols:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame()

# canonical columns for draws
DRAW_COLS = ["issue_id", "timestamp", "number", "color", "size"]

last_draws = load_df_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv", cols=DRAW_COLS)
predictions_df = load_df_from_r2_or_local(F_PREDICTIONS, cols=["created_at", "prediction", "pick_count"])
accuracy_log = load_df_from_r2_or_local(F_ACCURACY, cols=["timestamp", "predicted", "actual", "accuracy_pct"])

# ensure types
for df in (last_draws, predictions_df, accuracy_log):
    if df is None:
        df = pd.DataFrame()

# --------------------------
# Mapping rules (from your notes)
# - Numbers are single-digit 0..9
# - Size: 0-4 -> Small, 5-9 -> Big
# - Color:
#     - by default 0-4 -> Red, 5-9 -> Green
#     - special rule: numbers equal to 0 -> Red-purple, 5 -> Green-purple
#   (This is inferred from your note; adjust if needed)
# --------------------------
def infer_size_from_number(n):
    try:
        n = int(n)
    except:
        return ""
    return "Small" if 0 <= n <= 4 else "Big"

def infer_color_from_number(n):
    try:
        n = int(n)
    except:
        return ""
    if n == 0:
        return "Red-purple"
    if n == 5:
        return "Green-purple"
    return "Red" if 0 <= n <= 4 else "Green"

# --------------------------
# Prediction logic for 0..9 numbers
# - Simple frequency + recency penalty heuristic
# - pick_count defaults to 1 (you can choose more)
# --------------------------
def predict_next_numbers(df_draws, pick_count=1):
    # if no history, return empty
    if df_draws is None or df_draws.shape[0] == 0:
        return []
    # use numeric series of 'number' column
    nums = []
    for v in df_draws["number"].astype(str).values:
        try:
            nums.append(int(v))
        except:
            continue
    if not nums:
        return []
    freq = pd.Series(nums).value_counts().to_dict()  # number -> freq
    # recency: index of last occurrence (higher = more recent)
    last_index = {}
    for idx, v in enumerate(df_draws["number"].astype(str).values[::-1]):  # reverse for recency 0=most recent
        try:
            val = int(v)
            if val not in last_index:
                last_index[val] = idx
        except:
            continue
    # score = freq*(1.0) - recency*0.2
    scores = {}
    for n in range(0, 10):
        f = freq.get(n, 0)
        rec = last_index.get(n, 10000)
        scores[n] = f * 1.0 - rec * 0.2
    # sort by score desc
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picks = [n for n, s in sorted_nums[:pick_count]]
    return picks

# --------------------------
# Utilities: parse coinryze.org format lines
# Expected format:
# issue_id,timestamp,number,color,size
# e.g.
# 202510131045, 15:25:00 10/13/2025, 3, Green , Small
# --------------------------
def parse_line_to_row(line):
    parts = [p.strip() for p in line.split(",")]
    # allow when color or size has trailing spaces
    if len(parts) < 5:
        return None
    issue_id = parts[0]
    timestamp = parts[1]
    number = parts[2]
    color = parts[3]
    size = parts[4]
    # normalize
    return {
        "issue_id": issue_id,
        "timestamp": timestamp,
        "number": str(int(float(number))) if number != "" else "",
        "color": color,
        "size": size,
    }

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Coinryze Analyzer (issue_id format)", layout="wide")
st.title("Coinryze Analyzer — issue_id,timestamp,number,color,size")

# Top-level controls
col_top_a, col_top_b = st.columns([2,1])
with col_top_b:
    st.sidebar.header("Settings")
    pick_count = st.sidebar.number_input("Prediction size (how many numbers)", min_value=1, max_value=5, value=1)
    auto_predict_toggle = st.sidebar.checkbox("Auto-predict after adding a draw", value=True)
    show_debug = st.sidebar.checkbox("Show debug", value=False)

# Manual single-draw form
st.header("Manual Input — Add Single Last Draw")
with st.form("single_draw"):
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
    # infer missing color/size if not provided
    if number is None or number == "":
        st.error("Please provide a number 0-9.")
    else:
        if color == "" or size == "":
            inferred_color = infer_color_from_number(number)
            inferred_size = infer_size_from_number(number)
            if color == "":
                color = inferred_color
            if size == "":
                size = inferred_size
        new_row = {
            "issue_id": str(issue_id) if issue_id else datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            "timestamp": str(timestamp),
            "number": str(int(float(number))),
            "color": color,
            "size": size,
        }
        # append to top (most recent first)
        last_draws = pd.concat([pd.DataFrame([new_row]), last_draws], ignore_index=True)
        # persist to R2
        r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
        st.success("Draw added and synced to R2.")
        # if auto-predict, generate and show
        if auto_predict_toggle:
            picks = predict_next_numbers(last_draws, pick_count=pick_count)
            # prepare prediction record
            rec = {
                "created_at": datetime.datetime.utcnow().isoformat(),
                "prediction": json.dumps(picks),
                "pick_count": pick_count,
            }
            predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
            r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction: {picks}")
            # also log accuracy if you want to compare to previous draw (not done automatically)
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.write("Added. Scroll to see updated tables.")

st.write("---")
# Paste multiple lines (copy-paste from coinryze.org)
st.subheader("Paste lines from coinryze.org (one per line, same format)")
multiline = st.text_area("Paste CSV lines here (issue_id,timestamp,number,color,size)", height=120, placeholder="Paste rows like:\n202510131045, 15:25:00 10/13/2025, 3, Green , Small")
if st.button("Parse & Add Pasted Lines"):
    lines = [l.strip() for l in multiline.splitlines() if l.strip() != ""]
    added = 0
    for L in lines:
        parsed = parse_line_to_row(L)
        if parsed:
            # infer color/size if empty
            if parsed["color"] == "" or parsed["size"] == "":
                parsed["color"] = parsed["color"] or infer_color_from_number(parsed["number"])
                parsed["size"] = parsed["size"] or infer_size_from_number(parsed["number"])
            last_draws = pd.concat([pd.DataFrame([parsed]), last_draws], ignore_index=True)
            added += 1
    if added:
        r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Added {added} pasted rows and synced to R2.")
        if auto_predict_toggle:
            picks = predict_next_numbers(last_draws, pick_count=pick_count)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count}
            predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
            r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction: {picks}")
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else None
    else:
        st.warning("No valid rows parsed. Check format (issue_id,timestamp,number,color,size).")

st.write("---")
# Bulk file upload
st.subheader("Bulk CSV Upload (one or more CSV files)")
uploaded_files = st.file_uploader("Upload CSV(s) with columns issue_id,timestamp,number,color,size", accept_multiple_files=True, type=["csv"])
if uploaded_files:
    total_added = 0
    for f in uploaded_files:
        try:
            df_new = pd.read_csv(f, dtype=str)
            # try to normalize required columns
            if not set(["issue_id", "timestamp", "number"]).issubset(set([c.lower() for c in df_new.columns])):
                st.warning(f"File {f.name} missing required columns (issue_id,timestamp,number). Skipping.")
                continue
            # rename columns to canonical lower-case
            df_new.columns = [c.strip() for c in df_new.columns]
            # keep only required columns, try to map if case differs
            df_new2 = pd.DataFrame()
            for c in DRAW_COLS:
                # find matching column ignoring case
                match = next((orig for orig in df_new.columns if orig.lower() == c.lower()), None)
                if match:
                    df_new2[c] = df_new[match].astype(str)
                else:
                    df_new2[c] = ""
            # infer missing color/size
            for idx, row in df_new2.iterrows():
                if row["color"] == "" or row["size"] == "":
                    df_new2.at[idx, "color"] = row["color"] or infer_color_from_number(row["number"])
                    df_new2.at[idx, "size"] = row["size"] or infer_size_from_number(row["number"])
            last_draws = pd.concat([df_new2, last_draws], ignore_index=True)
            total_added += df_new2.shape[0]
        except Exception as e:
            st.error(f"Failed to read {f.name}: {e}")
    if total_added:
        r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
        st.success(f"Appended {total_added} rows from uploaded files and synced to R2.")
        if auto_predict_toggle:
            picks = predict_next_numbers(last_draws, pick_count=pick_count)
            rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count}
            predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
            r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
            st.info(f"Auto-prediction: {picks}")
        st.experimental_rerun() if hasattr(st, "experimental_rerun") else None

st.write("---")
# Historical draws (top 50)
st.subheader("Historical draws (top 50 shown)")
if last_draws is None or last_draws.shape[0] == 0:
    st.info("No historical draws yet. Add via manual input or upload.")
else:
    # show top 50
    st.dataframe(last_draws.head(50))

st.write("---")
# Smart Predictions section
st.subheader("Smart Predictions (based on history)")
if last_draws.shape[0] == 0:
    st.info("No history to predict from.")
else:
    picks = predict_next_numbers(last_draws, pick_count=pick_count)
    st.write("Predicted next numbers:", picks)
    # show inferred color/size for each predicted number
    pick_details = []
    for p in picks:
        pick_details.append({
            "number": p,
            "color": infer_color_from_number(p),
            "size": infer_size_from_number(p)
        })
    st.table(pd.DataFrame(pick_details))
    if st.button("Save this prediction"):
        rec = {"created_at": datetime.datetime.utcnow().isoformat(), "prediction": json.dumps(picks), "pick_count": pick_count}
        predictions_df = pd.concat([pd.DataFrame([rec]), predictions_df], ignore_index=True)
        r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
        st.success("Prediction saved to predictions.csv and synced to R2.")

st.write("---")
# Predictions history and accuracy utilities
st.subheader("Predictions history (top 20)")
if predictions_df is None or predictions_df.shape[0] == 0:
    st.info("No predictions yet.")
else:
    st.dataframe(predictions_df.head(20))
    if st.button("Download predictions.csv"):
        st.download_button("Download predictions.csv", data=predictions_df.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# Accuracy logging helper: compare latest saved prediction to latest actual draw
st.write("---")
st.subheader("Evaluate last prediction vs latest actual draw")
if predictions_df.shape[0] > 0 and last_draws.shape[0] > 0:
    last_pred = json.loads(predictions_df.iloc[0]["prediction"])
    latest_draw_num = None
    try:
        latest_draw_num = int(last_draws.iloc[0]["number"])
    except:
        latest_draw_num = None
    st.write("Last prediction:", last_pred)
    st.write("Latest actual draw (most recent):", latest_draw_num)
    if latest_draw_num is not None:
        overlap_pct = (len(set(last_pred).intersection({latest_draw_num})) / max(1, len(last_pred))) * 100.0
        st.write(f"Overlap with latest draw: {overlap_pct:.1f}%")
        if st.button("Log this accuracy"):
            rec = {"timestamp": datetime.datetime.utcnow().isoformat(), "predicted": json.dumps(last_pred), "actual": json.dumps([latest_draw_num]), "accuracy_pct": overlap_pct}
            accuracy_log = pd.concat([pd.DataFrame([rec]), accuracy_log], ignore_index=True)
            r2_put_bytes(F_ACCURACY, accuracy_log.to_csv(index=False).encode("utf-8"))
            st.success("Logged accuracy to accuracy_log.csv and synced to R2.")
else:
    st.info("Need at least one prediction and one actual draw to evaluate.")

st.write("---")
# Files in R2 and utilities
st.subheader("R2 Files & Utilities")
keys = r2_list_keys()
if keys:
    st.write("Files in bucket:")
    for k in keys:
        st.write("-", k)
        b = r2_get_bytes(k)
        if b is not None:
            st.download_button(f"Download {k}", data=BytesIO(b), file_name=k)
else:
    st.info("No files found in R2 bucket.")

col_u1, col_u2 = st.columns(2)
with col_u1:
    if st.button("Force sync all CSVs to R2"):
        r2_put_bytes(F_LAST_DRAWS, last_draws.to_csv(index=False).encode("utf-8"))
        r2_put_bytes(F_PREDICTIONS, predictions_df.to_csv(index=False).encode("utf-8"))
        r2_put_bytes(F_ACCURACY, accuracy_log.to_csv(index=False).encode("utf-8"))
        st.success("All CSVs uploaded to R2.")
with col_u2:
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

# debug
if show_debug:
    st.write("DEBUG INFO")
    st.write("Last draws rows:", last_draws.shape[0])
    st.write("Predictions rows:", predictions_df.shape[0])
    st.write("Accuracy rows:", accuracy_log.shape[0])
    st.write("R2 keys:", r2_list_keys())

st.markdown("""
## Quick how-to
- Paste exact lines from coinryze.org into the paste box or add a single row using the form above.
- Format must be: `issue_id,timestamp,number,color,size`
- Example:
