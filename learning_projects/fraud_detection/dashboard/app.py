import streamlit as st
import psycopg2
import pandas as pd
import os
import time

# Config
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("🛡️ Real-Time AI Fraud Detection")

def get_data():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        query = "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame()

# Layout
col1, col2, col3 = st.columns(3)
placeholder = st.empty()

while True:
    df = get_data()
    
    if not df.empty:
        total_tx = len(df)
        anomalies = df['is_anomaly'].sum()
        avg_score = df['fraud_score'].mean()
        
        with col1:
            st.metric("Transactions (Last 100)", total_tx)
        with col2:
            st.metric("Detected Anomalies", int(anomalies), delta_color="inverse")
        with col3:
            st.metric("Avg Risk Score", f"{avg_score:.4f}")

        with placeholder.container():
            st.subheader("Live Feed")
            
            # Scatter Plot
            st.scatter_chart(df, x='timestamp', y='fraud_score', color='is_anomaly')
            
            # Data Table
            st.dataframe(df)
            
    else:
        st.warning("Waiting for data...")
    
    time.sleep(2)
