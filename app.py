# app.py
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import pandas as pd

# --------------------------
# R2 Configuration
# --------------------------
R2_KEY_ID = "YOUR_R2_ACCESS_KEY_ID"
R2_SECRET = "YOUR_R2_SECRET_ACCESS_KEY"
R2_BUCKET = "YOUR_BUCKET_NAME"
R2_ENDPOINT = "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"  # Actual endpoint

# Create S3 client for R2
s3 = boto3.client(
    's3',
    region_name='auto',       # Must be 'auto' for R2
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY_ID,
    aws_secret_access_key=R2_SECRET
)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Coinryze Analyzer with R2", layout="wide")
st.title("Coinryze Analyzer + Cloudflare R2")

# --- Upload file to R2 ---
st.header("Upload a file to R2")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    try:
        s3.put_object(
            Bucket=R2_BUCKET,
            Key=uploaded_file.name,
            Body=uploaded_file.getbuffer()
        )
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    except ClientError as e:
        st.error(f"Error uploading file: {e}")

# --- List files in R2 ---
st.header("Files in R2 Bucket")
try:
    response = s3.list_objects_v2(Bucket=R2_BUCKET)
    files = response.get("Contents", [])
    if files:
        for obj in files:
            st.write(f"- {obj['Key']} (size: {obj['Size']} bytes)")
    else:
        st.info("Bucket is empty.")
except ClientError as e:
    st.error(f"Error listing files: {e}")

# --- Load seed CSV or example data (previous app) ---
st.header("Coinryze Data Table")
# Example: load a local CSV or R2 CSV
try:
    # If you have a seed CSV locally
    seed_df = pd.read_csv("backend/data/seed.csv")
    st.dataframe(seed_df)
except FileNotFoundError:
    st.warning("Seed CSV not found. You can upload a CSV to the R2 bucket.")

# --- Additional dashboard controls (from previous app) ---
st.header("Dashboard Controls")
refresh_button = st.button("Refresh Data")
if refresh_button:
    st.experimental_rerun()  # Simple refresh to reload data

st.markdown("""
> This dashboard now supports:
> - Uploading files to Cloudflare R2
> - Listing files in your R2 bucket
> - Loading seed CSV data for your Coinryze Analyzer
""")
