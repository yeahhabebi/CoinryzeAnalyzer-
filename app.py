# app.py
import streamlit as st
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from io import BytesIO

# --------------------------
# R2 Configuration
# --------------------------
R2_KEY_ID = "7423969d6d623afd9ae23258a6cd2839"
R2_SECRET = "dd858bf600c0d8e63cd047d128b46ad6df0427daef29f57c312530da322fc63c"
R2_BUCKET = "coinryze-analyzer"
R2_ENDPOINT = "https://6d266c53f2f03219a25de8f12c50bc3b.r2.cloudflarestorage.com"

# Create S3 client for R2
s3 = boto3.client(
    's3',
    region_name='auto',
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
uploaded_file = st.file_uploader("Choose a file", type=None)
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
files = []
try:
    response = s3.list_objects_v2(Bucket=R2_BUCKET)
    files = response.get("Contents", [])
    if files:
        for obj in files:
            file_name = obj['Key']
            file_size = obj['Size']
            st.write(f"- {file_name} (size: {file_size} bytes)")
            
            # Download button for each file
            try:
                file_obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
                file_bytes = file_obj['Body'].read()
                st.download_button(
                    label=f"Download {file_name}",
                    data=BytesIO(file_bytes),
                    file_name=file_name
                )
            except ClientError as e:
                st.error(f"Error downloading file {file_name}: {e}")
    else:
        st.info("Bucket is empty.")
except ClientError as e:
    st.error(f"Error listing files: {e}")

# --- Load seed CSV or example data ---
st.header("Coinryze Data Table")
seed_df = None
try:
    seed_df = pd.read_csv("backend/data/seed.csv")
except FileNotFoundError:
    st.warning("Seed CSV not found. You can upload a CSV to the R2 bucket.")

if seed_df is not None:
    st.dataframe(seed_df)

# --- Refresh data button ---
st.header("Dashboard Controls")
if st.button("Refresh Data"):
    # Simply reload data
    if seed_df is not None:
        st.dataframe(seed_df)
    # Re-list files in R2
    try:
        response = s3.list_objects_v2(Bucket=R2_BUCKET)
        files = response.get("Contents", [])
        if files:
            st.write("Files in R2 bucket refreshed.")
        else:
            st.info("Bucket is empty after refresh.")
    except ClientError as e:
        st.error(f"Error refreshing files: {e}")

st.markdown("""
> Features:
> - Upload files to Cloudflare R2
> - List and download files from R2
> - Display seed CSV data
> - Refresh data manually
""")
