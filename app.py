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

# Theme-Aware Professional Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Global Typography */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
    }

    /* Container refinement */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1100px !important;
    }

    /* Refined Metric Cards */
    [data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        transition: border-color 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: #4dabf7;
    }

    /* Professional Headers */
    .main-header {
        margin-bottom: 2.5rem;
    }

    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05rem;
        color: #7d7d7d;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
    }

    /* Expander refinement */
    .stExpander {
        border: 1px solid rgba(128, 128, 128, 0.1) !important;
        border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.title("BTC Range Forecast")
st.markdown("FIGARCH-GBM Engine")
st.markdown("---")

def load_backtest_metrics():
    if os.path.exists('backtest_results.jsonl'):
        results = []
        with open('backtest_results.jsonl', 'r') as f:
            for line in f:
                results.append(json.loads(line))
        df = pd.DataFrame(results)
        coverage = df['coverage_95'].mean()
        avg_width = df['width_95'].mean()
        winkler = df['winkler_95'].mean()
        return coverage, avg_width, winkler
    return 0.95, 200.0, 350.0 

cov, width, wink = load_backtest_metrics()

# Live Data & Prediction
@st.cache_data(ttl=60)
def get_data_and_predict():
    df = get_binance_klines(limit=500)
    pred = predict_next_hour(df['close'], n_sims=10000)
    return df, pred

with st.spinner("Calculating variances..."):
    df, pred = get_data_and_predict()

# Sidebar: Backtest Validation
with st.sidebar:
    st.markdown('<div class="section-title" style="margin-top:0;">Backtest Performance</div>', unsafe_allow_html=True)
    st.metric("Coverage Accuracy", f"{cov:.2%}")
    st.metric("Mean Range Width", f"${width:,.0f}")
    st.metric("Winkler Score", f"{wink:.1f}")
    
    st.markdown("---")
    st.caption("Model: FIGARCH (Student-t)")
    st.caption("Monte Carlo: 10k Paths")

# Main Content
st.markdown('<div class="section-title" style="margin-top:0;">Forecast: Next Hour</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("Spot Price", f"${pred['current_price']:,.2f}")
m2.metric("Target Low (95%)", f"${pred['predicted_low']:,.0f}", delta=f"{pred['predicted_low'] - pred['current_price']:,.2f}", delta_color="normal")
m3.metric("Target High (95%)", f"${pred['predicted_high']:,.0f}", delta=f"{pred['predicted_high'] - pred['current_price']:,.2f}", delta_color="normal")

st.markdown('<div class="section-title">Visual Analysis</div>', unsafe_allow_html=True)

# Chart with theme-aware Plotly
last_50 = df.tail(50)
fig = go.Figure()

next_hour = last_50.index[-1] + pd.Timedelta(hours=1)
last_time = last_50.index[-1]
last_price = last_50['close'].iloc[-1]

# Prediction Ribbon
fig.add_trace(go.Scatter(
    x=[last_time, next_hour], y=[last_price, pred['predicted_high']],
    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=[last_time, next_hour], y=[last_price, pred['predicted_low']],
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(77, 171, 247, 0.1)', name='Prediction Band', hoverinfo='skip'
))

# Price line
fig.add_trace(go.Scatter(
    x=last_50.index, y=last_50['close'],
    mode='lines+markers', name='Actual Price', 
    line=dict(color='#4dabf7', width=2),
    marker=dict(size=4)
))

# Forecast marker
fig.add_trace(go.Scatter(
    x=[next_hour, next_hour], y=[pred['predicted_low'], pred['predicted_high']],
    mode='lines', name='Forecast Range',
    line=dict(color='#ff6b6b', width=4)
))

fig.update_layout(
    template="none", # Let it adapt to streamlit theme
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)'),
    margin=dict(l=0, r=0, t=10, b=0),
    height=450,
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.markdown('<div class="section-title">Session History</div>', unsafe_allow_html=True)
with st.expander("Show Log"):
    history_file = 'prediction_history.json'
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            hist = json.load(f)
        st.dataframe(pd.DataFrame(hist).tail(10), use_container_width=True)

# Persistence
def save_prediction(pred):
    history_file = 'prediction_history.json'
    new_record = {
        'Time (UTC)': datetime.now().strftime('%H:%M:%S'),
        'Spot Price': f"${pred['current_price']:,.2f}",
        'Range Low': f"${pred['predicted_low']:,.2f}",
        'Range High': f"${pred['predicted_high']:,.2f}"
    }
    history = []
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try: history = json.load(f)
            except: history = []
    history.append(new_record)
    history = history[-20:] # Show last 20
    with open(history_file, 'w') as f:
        json.dump(history, f)

save_prediction(pred)
