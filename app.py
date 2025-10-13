import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import boto3

# ===========================
# R2 Cloudflare Config
# ===========================
R2_KEY_ID = "YOUR_R2_KEY_ID"
R2_SECRET = "YOUR_R2_SECRET"
R2_BUCKET = "YOUR_BUCKET_NAME"
R2_ENDPOINT = "https://<account_id>.r2.cloudflarestorage.com"

s3 = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET
)

# ===========================
# Helper Functions
# ===========================
def load_csv(filename):
    try:
        obj = s3.get_object(Bucket=R2_BUCKET, Key=filename)
        df = pd.read_csv(obj['Body'])
        return df
    except:
        return pd.DataFrame(columns=['issue_id', 'timestamp', 'number', 'color', 'size', 'odd_even'])

def save_csv(df, filename):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=R2_BUCKET, Key=filename, Body=csv_buffer.getvalue())

def generate_predictions(df_last, n_predictions=5):
    if df_last.empty:
        return pd.DataFrame(columns=df_last.columns)
    
    preds = []
    last_row = df_last.iloc[-1]
    for i in range(n_predictions):
        next_number = (last_row['number'] + i + 1) % 37
        next_color = "Red" if (last_row['color'] == "Black") else "Black"
        next_size = "Small" if (last_row['size'] == "Big") else "Big"
        next_odd_even = "Odd" if (last_row['odd_even'] == "Even") else "Even"
        next_issue_id = str(int(last_row['issue_id']) + i + 1)
        preds.append({
            "issue_id": next_issue_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number": next_number,
            "color": next_color,
            "size": next_size,
            "odd_even": next_odd_even
        })
    return pd.DataFrame(preds)

# ===========================
# Streamlit UI
# ===========================
st.title("CoinryzeAnalyzer Dashboard 🔮")

# --- Bulk Input ---
st.subheader("Add Last Draw Results (Bulk)")
uploaded_file = st.file_uploader("Upload CSV with last draws", type=["csv"])
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    df_last = load_csv("last_draws.csv")
    df_last = pd.concat([df_last, df_upload], ignore_index=True)
    save_csv(df_last, "last_draws.csv")
    st.success(f"{len(df_upload)} rows added to last_draws.csv!")

# --- Manual Single Input ---
st.subheader("Add Single Last Draw")
with st.form("single_draw_form"):
    issue_id = st.text_input("Issue ID")
    number = st.number_input("Number", min_value=0, max_value=36)
    color = st.selectbox("Color", ["Red", "Black", "Green"])
    size = st.selectbox("Size", ["Small", "Big"])
    odd_even = st.selectbox("Odd/Even", ["Odd", "Even"])
    submitted = st.form_submit_button("Add Draw")
    
if submitted:
    df_last = load_csv("last_draws.csv")
    new_row = {
        "issue_id": issue_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "number": number,
        "color": color,
        "size": size,
        "odd_even": odd_even
    }
    df_last = pd.concat([df_last, pd.DataFrame([new_row])], ignore_index=True)
    save_csv(df_last, "last_draws.csv")
    st.success("Single draw added!")

# --- Display Last Draws ---
st.subheader("Last Draws Table")
df_last = load_csv("last_draws.csv")
st.dataframe(df_last)

# --- Predictions ---
st.subheader("Generate Next Predictions")
num_pred = st.number_input("Number of predictions", min_value=1, max_value=20, value=5)
if st.button("Generate Predictions"):
    df_pred = generate_predictions(df_last, n_predictions=num_pred)
    save_csv(df_pred, "predictions.csv")
    st.success(f"{len(df_pred)} predicted outcomes generated!")
    st.dataframe(df_pred)
