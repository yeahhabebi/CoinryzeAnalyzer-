# app.py
"""
Coinryze Analyzer - Full Streamlit dashboard with Cloudflare R2 integration.

Features:
- Manual Input Form (add single draw)
- Bulk CSV Upload (append many past draws)
- Smart Predictions (frequency + streak heuristics)
- Auto-Refresh (set interval in sidebar)
- Sync CSVs to Cloudflare R2: last_draws.csv, predictions.csv, accuracy_log.csv
- Accuracy tracking and chart
"""

import streamlit as st
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError
from io import BytesIO
import altair as alt
import datetime
import json
import html

# --------------------------
# R2 Configuration (already provided)
# --------------------------
R2_KEY_ID = "7423969d6d623afd9ae23258a6cd2839"
R2_SECRET = "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c"
R2_BUCKET = "coinryze-analyzer"
R2_ENDPOINT = "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"

# Filenames used in the app
F_LAST_DRAWS = "last_draws.csv"        # historical draws (append here)
F_PREDICTIONS = "predictions.csv"      # predictions history
F_ACCURACY = "accuracy_log.csv"        # accuracy history

# Create S3/R2 client
s3 = boto3.client(
    's3',
    region_name='auto',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET
)

# --------------------------
# Helper functions for R2
# --------------------------
def r2_get_object_bytes(key):
    """Return bytes of object from R2 or None if not exists"""
    try:
        resp = s3.get_object(Bucket=R2_BUCKET, Key=key)
        return resp['Body'].read()
    except ClientError as e:
        #  NoSuchKey or other access errors
        return None

def r2_put_bytes(key, bts, content_type="text/csv"):
    """Put bytes to R2. Returns True if OK."""
    try:
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=bts, ContentType=content_type)
        return True
    except ClientError as e:
        st.error(f"R2 upload error for {key}: {e}")
        return False

def r2_list_keys(prefix=""):
    try:
        resp = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix=prefix)
        return [o['Key'] for o in resp.get('Contents', [])]
    except ClientError:
        return []

# --------------------------
# Data loading/saving helpers
# --------------------------
def load_csv_from_r2_or_local(key, local_path=None):
    """Try to load CSV from R2; if not available, fallback to local path (if given)."""
    b = r2_get_object_bytes(key)
    if b is not None:
        return pd.read_csv(BytesIO(b))
    if local_path:
        try:
            return pd.read_csv(local_path)
        except FileNotFoundError:
            return None
    return None

def save_df_to_r2(df, key):
    b = df.to_csv(index=False).encode('utf-8')
    return r2_put_bytes(key, b)

# --------------------------
# Smart prediction algorithm (simple heuristic)
# - Use frequency of numbers in historical draws
# - Prefer numbers that are 'hot' and avoid very recent repeats (basic streak handling)
# NOTE: Adjust to your draw format (here we assume draws are sets of integers in columns 'n1'..'n6')
# --------------------------
def simple_predict(df_last_draws, pick_count=6):
    """
    df_last_draws: DataFrame with columns ['date','n1','n2',...]
    returns list of predicted numbers
    """
    # Flatten number columns
    num_cols = [c for c in df_last_draws.columns if c.startswith('n')]
    if not num_cols:
        return []

    numbers = df_last_draws[num_cols].values.flatten()
    numbers = numbers[~pd.isna(numbers)].astype(int)
    freq = pd.Series(numbers).value_counts().sort_values(ascending=False)
    # Basic streak detection: compute last occurrence (index) for each number (higher index == more recent)
    last_occ = {}
    for idx, row in df_last_draws.reset_index().iterrows():
        for c in num_cols:
            val = row.get(c)
            if pd.notna(val):
                last_occ[int(val)] = idx  # last seen at idx

    # Score = frequency * weight - recency_penalty
    scores = {}
    for num, f in freq.items():
        recency = last_occ.get(int(num), 1e6)
        score = f * 2.0 - (recency * 0.01)  # tweak weights as desired
        scores[int(num)] = score

    # Include numbers not yet seen with small base score, to allow novelty
    all_candidate_nums = range(1, 51)  # assume numbers 1..50 (adjust if different)
    for n in all_candidate_nums:
        if n not in scores:
            scores[n] = 0.1

    # Pick top `pick_count` numbers by score
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    picked = [n for n, s in sorted_nums[:pick_count]]
    return picked

# --------------------------
# Accuracy calculation:
# Compare predictions.csv last prediction vs actual last draw (exact match rate or overlap rate)
# We'll compute overlap percentage (# predicted numbers that appear in actual draw / pick_count)
# --------------------------
def compute_accuracy(pred_numbers, actual_numbers):
    if not pred_numbers or not actual_numbers:
        return 0.0
    pred_set = set(pred_numbers)
    actual_set = set(actual_numbers)
    overlap = len(pred_set.intersection(actual_set))
    return float(overlap) / float(len(pred_numbers)) * 100.0

# --------------------------
# App UI & logic
# --------------------------
st.set_page_config(page_title="Coinryze Analyzer + R2", layout="wide")
st.title("Coinryze Analyzer + Cloudflare R2 (Full Dashboard)")

# Sidebar controls
st.sidebar.header("Settings")
auto_refresh_secs = st.sidebar.slider("Auto-refresh interval (seconds, 0 = off)", 0, 300, 0, step=5)
pick_count = st.sidebar.number_input("Prediction size (how many numbers to predict)", min_value=1, max_value=10, value=6)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

# Auto-refresh via injected JavaScript (works regardless of Streamlit version)
if auto_refresh_secs and auto_refresh_secs > 0:
    # Use meta refresh via JS to reload page after N seconds
    js = f"""
    <script>
      // only reload if page is visible (avoid looping while hidden)
      setTimeout(function() {{
        if (document.visibilityState === 'visible') {{
          location.reload();
        }}
      }}, {int(auto_refresh_secs * 1000)});
    </script>
    """
    st.sidebar.markdown(f"Auto-refresh enabled every **{auto_refresh_secs}**s")
    st.components.v1.html(js)

# Load existing dataframes
last_draws = load_csv_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv")
predictions_df = load_csv_from_r2_or_local(F_PREDICTIONS)
accuracy_log = load_csv_from_r2_or_local(F_ACCURACY)

# Ensure structure if None
if last_draws is None:
    last_draws = pd.DataFrame(columns=['date','n1','n2','n3','n4','n5','n6'])
if predictions_df is None:
    predictions_df = pd.DataFrame(columns=['created_at','prediction','pick_count'])
if accuracy_log is None:
    accuracy_log = pd.DataFrame(columns=['timestamp','predicted','actual','accuracy_pct'])

# Layout: two columns top
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Manual Input — Add Single Last Draw")
    with st.form("single_draw_form", clear_on_submit=True):
        # Accept date and numbers (comma separated)
        date_val = st.date_input("Draw date", value=datetime.date.today())
        nums_text = st.text_input("Draw numbers (comma separated)", placeholder="e.g. 3,8,15,22,27,34")
        submitted = st.form_submit_button("Add draw")
        if submitted:
            try:
                nums = [int(x.strip()) for x in nums_text.split(",") if x.strip()!='']
                if len(nums) != pick_count:
                    st.warning(f"You entered {len(nums)} numbers but prediction size in sidebar is {pick_count}. App will still store as provided.")
                row = {'date': pd.to_datetime(date_val).strftime("%Y-%m-%d")}
                for i in range(1, 7):
                    row[f"n{i}"] = nums[i-1] if i-1 < len(nums) else np.nan
                last_draws = pd.concat([pd.DataFrame([row]), last_draws], ignore_index=True)
                st.success("Draw added locally.")
                # Save to R2
                save_df_to_r2(last_draws, F_LAST_DRAWS)
                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.info("Refresh to see changes.")
            except Exception as e:
                st.error(f"Error parsing numbers: {e}")

    st.write("---")
    st.subheader("Bulk CSV Upload (append historical draws)")
    st.markdown("Upload a CSV where columns include at least `date` and `n1`..`n6` (or similar).")
    bulk_files = st.file_uploader("Upload CSV file(s)", type=["csv"], accept_multiple_files=True)
    if bulk_files:
        appended = 0
        for bf in bulk_files:
            try:
                df_new = pd.read_csv(bf)
                # Try to normalize columns: keep date and n* columns; if extra columns exist ignore
                cols = [c for c in df_new.columns if c.lower().startswith('n') or c.lower()=='date']
                if 'date' not in [c.lower() for c in df_new.columns]:
                    st.warning(f"File {bf.name} missing 'date' column — skipping.")
                    continue
                df_new = df_new[cols]
                # Append to existing last_draws (newest first)
                last_draws = pd.concat([df_new, last_draws], ignore_index=True).drop_duplicates().reset_index(drop=True)
                appended += len(df_new)
            except Exception as e:
                st.error(f"Failed to process {bf.name}: {e}")
        if appended:
            save_df_to_r2(last_draws, F_LAST_DRAWS)
            st.success(f"Appended {appended} rows to last_draws and synced to R2.")
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

    st.write("---")
    st.subheader("Historical draws (top 50 shown)")
    st.dataframe(last_draws.head(50))

with col2:
    st.subheader("Smart Predictions")
    if last_draws.empty or last_draws.shape[0]==0:
        st.info("No historical draws available — add draws first (manual or CSV).")
    else:
        # Ensure we have n1..n6 columns; if dataset has different pick_count, adapt
        # We'll rely on simple_predict which expects n1..n6 columns; create them if missing
        # Normalize column names
        expected_ncols = [f"n{i}" for i in range(1, pick_count+1)]
        # If last_draws doesn't have expected columns but has any n* columns - ok
        pred = simple_predict(last_draws, pick_count=pick_count)
        st.markdown(f"**Prediction ({pick_count} numbers):** `{', '.join(map(str,pred))}`")
        # Save prediction to history (predictions_df)
        if st.button("Save This Prediction"):
            now = datetime.datetime.utcnow().isoformat()
            predictions_df = pd.concat([pd.DataFrame([{
                'created_at': now,
                'prediction': json.dumps(pred),
                'pick_count': pick_count
            }]), predictions_df], ignore_index=True)
            save_df_to_r2(predictions_df, F_PREDICTIONS)
            st.success("Prediction saved and synced to R2.")
        # Also allow immediate quick-evaluate against latest actual draw
        latest_draw_row = last_draws.iloc[0] if last_draws.shape[0]>0 else None
        if latest_draw_row is not None:
            actual_nums = []
            for c in last_draws.columns:
                if c.startswith('n'):
                    v = latest_draw_row.get(c)
                    if pd.notna(v):
                        actual_nums.append(int(v))
            if actual_nums:
                acc_pct = compute_accuracy(pred, actual_nums)
                st.write(f"Overlap with latest draw ({latest_draw_row.get('date')}): **{acc_pct:.1f}%**")
                if st.button("Log this accuracy to accuracy_log"):
                    timestamp = datetime.datetime.utcnow().isoformat()
                    accuracy_log = pd.concat([pd.DataFrame([{
                        'timestamp': timestamp,
                        'predicted': json.dumps(pred),
                        'actual': json.dumps(actual_nums),
                        'accuracy_pct': acc_pct
                    }]), accuracy_log], ignore_index=True)
                    save_df_to_r2(accuracy_log, F_ACCURACY)
                    st.success("Logged accuracy and synced to R2.")

    st.write("---")
    st.subheader("Files in R2 bucket")
    try:
        keys = r2_list_keys()
        if keys:
            for k in keys:
                st.write(f"- {k}")
                # Provide download link/button
                b = r2_get_object_bytes(k)
                if b is not None:
                    st.download_button(label=f"Download {k}", data=BytesIO(b), file_name=k)
        else:
            st.info("Bucket is empty (no keys found).")
    except Exception as e:
        st.error(f"Error listing R2 bucket: {e}")

# --- Accuracy tracking and trend chart ---
st.markdown("---")
st.subheader("Accuracy Tracker & Trend")

if not accuracy_log.empty:
    # Prepare a time series
    try:
        accuracy_log['timestamp'] = pd.to_datetime(accuracy_log['timestamp'])
        df_acc = accuracy_log.sort_values('timestamp', ascending=True)
        st.write("Recent accuracy logs:")
        st.dataframe(df_acc.head(50))
        # Altair line chart
        chart = alt.Chart(df_acc).mark_line(point=True).encode(
            x='timestamp:T',
            y='accuracy_pct:Q',
            tooltip=['timestamp:T','accuracy_pct:Q']
        ).properties(width=800, height=300, title='Accuracy over time')
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render accuracy chart: {e}")
else:
    st.info("No accuracy logs yet. Log accuracy from the Smart Predictions pane after evaluating against a real draw.")

# --- Buttons for manual sync and housekeeping ---
st.markdown("---")
st.subheader("Utilities")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Sync all local dataframes to R2 (force)"):
        ok1 = save_df_to_r2(last_draws, F_LAST_DRAWS)
        ok2 = save_df_to_r2(predictions_df, F_PREDICTIONS)
        ok3 = save_df_to_r2(accuracy_log, F_ACCURACY)
        if ok1 and ok2 and ok3:
            st.success("All files synced to R2.")
        else:
            st.warning("Some files may not have synced; check logs above.")

with col_b:
    if st.button("Reload data from R2 (force)"):
        # Force reload
        ld = load_csv_from_r2_or_local(F_LAST_DRAWS, local_path="backend/data/seed.csv")
        pdx = load_csv_from_r2_or_local(F_PREDICTIONS)
        agr = load_csv_from_r2_or_local(F_ACCURACY)
        if ld is not None:
            last_draws = ld
        if pdx is not None:
            predictions_df = pdx
        if agr is not None:
            accuracy_log = agr
        st.success("Reloaded available files from R2. Please scroll to sections to see updates.")

# Debug info
if show_debug:
    st.markdown("**DEBUG**")
    st.write(f"R2 endpoint: {R2_ENDPOINT}")
    st.write(f"Bucket: {R2_BUCKET}")
    st.write("Keys in bucket:", r2_list_keys())
    st.write("Last draws shape:", last_draws.shape)
    st.write("Predictions shape:", predictions_df.shape)
    st.write("Accuracy log shape:", accuracy_log.shape)

st.markdown("## Notes / How to use")
st.markdown("""
**Quick guide**
1. **Add draws**
   - Use *Manual Input* to add a single latest draw (enter exactly comma-separated numbers).
   - Or upload historical CSV(s) in *Bulk CSV Upload*. CSV should include `date` and `n1..n6` (or similar).
2. **Make predictions**
   - Sidebar: set `Prediction size` (default 6).
   - Smart Predictions box shows suggested numbers (based on historical frequency & simple streak penalty).
   - Click *Save This Prediction* to record it in `predictions.csv` and sync to R2.
   - Evaluate vs latest actual draw: click *Log this accuracy* to append to `accuracy_log.csv`.
3. **Auto-refresh**
   - In sidebar set Auto-refresh interval >0 to enable periodic page reloads.
4. **Syncing**
   - Files auto-save on add/upload actions; use `Sync all local dataframes to R2` to force push.
   - Use `Reload data from R2` to pull latest files from R2.
5. **Files stored in R2**
   - last_draws.csv — historical draws
   - predictions.csv — prediction history
   - accuracy_log.csv — logged accuracy entries

**CSV format expectations**
- `last_draws.csv`: columns: `date`, `n1`, `n2`, ... up to number of picks (e.g. n6)
- `predictions.csv`: created automatically (fields: created_at, prediction (json array), pick_count)
- `accuracy_log.csv`: created automatically (timestamp, predicted (json array), actual (json array), accuracy_pct)

**Security**
- Consider moving R2 keys into environment variables or Streamlit secrets for production.
""")
