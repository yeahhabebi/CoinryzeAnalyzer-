# app.py
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import pandas as pd

# --------------------------
# R2 Configuration
# --------------------------
R2_KEY_ID = "7423969d6d623afd9ae23258a6cd2839"         # Your R2 Access Key ID
R2_SECRET = "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c"     # Your R2 Secret Access Key
R2_BUCKET = "coinryze-analyzer"                       # Your bucket name
R2_ENDPOINT = "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"

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

# --- Load seed CSV or example data ---
st.header("Coinryze Data Table")
try:
    # If you have a seed CSV locally
    seed_df = pd.read_csv("backend/data/seed.csv")
    st.dataframe(seed_df)
except FileNotFoundError:
    st.warning("Seed CSV not found. You can upload a CSV to the R2 bucket.")

# --- Additional dashboard controls ---
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
