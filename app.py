import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os
import json
from data_utils import get_binance_klines
from model import predict_next_hour

# Page config
st.set_page_config(page_title="BTC Analytics", layout="wide")

# Charcoal Minimal Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    /* Container refinement */
    .block-container {
        padding-top: 3rem !important;
        max-width: 1000px !important;
    }

    /* Minimal Metric Styling (Theme Responsive) */
    [data-testid="stMetric"] {
        background: none;
        border: none;
        padding: 0;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        opacity: 0.6;
    }

    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 500 !important;
    }

    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 2.5rem;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        opacity: 0.4;
        margin-bottom: 1.5rem;
        margin-top: 3.5rem;
        letter-spacing: 0.12rem;
    }

    /* Sidebar minimal adjustment */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }

    hr {
        opacity: 0.1;
    }
    </style>
    """, unsafe_allow_html=True)

def load_backtest_metrics():
    if os.path.exists('backtest_results.jsonl'):
        results = []
        with open('backtest_results.jsonl', 'r') as f:
            for line in f:
                results.append(json.loads(line))
        df = pd.DataFrame(results)
        return df['coverage_95'].mean(), df['width_95'].mean()
    return 0.95, 200.0

cov, width = load_backtest_metrics()

@st.cache_data(ttl=60)
def get_data():
    df = get_binance_klines(limit=500)
    pred = predict_next_hour(df['close'])
    return df, pred

df, pred = get_data()

# Layout
st.markdown('<div class="main-title">BTC Forecast</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-label">Model Accuracy</div>', unsafe_allow_html=True)
    st.metric("Coverage", f"{cov:.1%}")
    st.metric("Range", f"${width:,.0f}")
    st.markdown("---")
    st.caption("FIGARCH-GBM Engine")

# Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Spot", f"${pred['current_price']:,.2f}")
m2.metric("Low", f"${pred['predicted_low']:,.0f}")
m3.metric("High", f"${pred['predicted_high']:,.0f}")

st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)

# Plotly Minimal
last_50 = df.tail(50)
fig = go.Figure()

next_hour = last_50.index[-1] + pd.Timedelta(hours=1)
last_time = last_50.index[-1]
last_price = last_50['close'].iloc[-1]

# Prediction Band
fig.add_trace(go.Scatter(
    x=[last_time, next_hour], y=[pred['predicted_high'], pred['predicted_high']],
    mode='lines', line=dict(color='#333', width=1, dash='dot'), showlegend=False
))
fig.add_trace(go.Scatter(
    x=[last_time, next_hour], y=[pred['predicted_low'], pred['predicted_low']],
    mode='lines', line=dict(color='#333', width=1, dash='dot'), showlegend=False
))

# Price line
fig.add_trace(go.Scatter(
    x=last_50.index, y=last_50['close'],
    mode='lines', name='Price', 
    line=dict(color='#4dabf7', width=1.5)
))

# Forecast Range
fig.add_trace(go.Scatter(
    x=[next_hour, next_hour], y=[pred['predicted_low'], pred['predicted_high']],
    mode='lines', name='Forecast',
    line=dict(color='#fff', width=2)
))

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showgrid=False, 
        zeroline=False,
    ),
    yaxis=dict(
        showgrid=True, 
        gridcolor='rgba(128,128,128,0.1)', 
        zeroline=False,
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    height=400,
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# Log
st.markdown('<div class="section-label">Session Intelligence</div>', unsafe_allow_html=True)
history_file = 'prediction_history.json'
if os.path.exists(history_file):
    with open(history_file, 'r') as f:
        hist = json.load(f)
    df_hist = pd.DataFrame(hist).tail(10)
    # Ensure columns are in order
    cols = ['Time (UTC)', 'Spot Price', 'Range Low', 'Range High']
    # Filter to only show records with all matching columns to avoid NaN fragmentation
    df_hist = df_hist[[c for c in cols if c in df_hist.columns]]
    st.dataframe(df_hist, use_container_width=True)

# Persistence
def save_prediction(pred):
    new_record = {
        'Time (UTC)': datetime.now().strftime('%H:%M:%S'),
        'Spot Price': f"${pred['current_price']:,.2f}",
        'Range Low': f"${pred['predicted_low']:,.0f}",
        'Range High': f"${pred['predicted_high']:,.0f}"
    }
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try: history = json.load(f)
            except: history = []
    history.append(new_record)
    with open(history_file, 'w') as f:
        json.dump(history[-20:], f)

save_prediction(pred)
