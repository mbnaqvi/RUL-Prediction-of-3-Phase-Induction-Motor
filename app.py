import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import warnings
import os

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Motor RUL Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

header[data-testid="stHeader"] {
    background-color: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
    display: none !important;
}

div[data-testid="stToolbar"] {
    display: none !important;
}

div[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu {
    display: none !important;
}

footer {
    display: none !important;
}

section[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}

section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}

.sticky-nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background-color: #0d1117;
    border-bottom: 1px solid #21262d;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 52px;
}

.nav-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #e6edf3;
    letter-spacing: -0.01em;
}

.nav-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7d8590;
    margin-left: 16px;
}

.nav-badge {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
}

.nav-right {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-stat {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7d8590;
    padding: 3px 10px;
    border: 1px solid #21262d;
    border-radius: 4px;
    background: #161b22;
}

.main-content-offset {
    margin-top: 60px;
}

.sidebar-toggle-btn {
    position: fixed;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    z-index: 9998;
    background: #161b22;
    border: 1px solid #30363d;
    border-left: none;
    border-radius: 0 6px 6px 0;
    color: #58a6ff;
    font-size: 1rem;
    padding: 10px 6px;
    cursor: pointer;
    writing-mode: vertical-rl;
    text-orientation: mixed;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #58a6ff;
    transition: background 0.2s;
}

.metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    line-height: 1.1;
    margin: 6px 0 2px 0;
}

.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7d8590;
    margin-bottom: 4px;
}

.metric-sub {
    font-size: 0.78rem;
    color: #7d8590;
    font-family: 'DM Mono', monospace;
}

.alert-critical {
    background: #1a0a0a;
    border: 1px solid #f85149;
    border-left: 4px solid #f85149;
    border-radius: 6px;
    padding: 12px 16px;
    color: #f85149;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 8px 0;
    letter-spacing: 0.02em;
}

.alert-warning {
    background: #1a1200;
    border: 1px solid #d29922;
    border-left: 4px solid #d29922;
    border-radius: 6px;
    padding: 12px 16px;
    color: #d29922;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 8px 0;
    letter-spacing: 0.02em;
}

.alert-ok {
    background: #0a1a0f;
    border: 1px solid #3fb950;
    border-left: 4px solid #3fb950;
    border-radius: 6px;
    padding: 12px 16px;
    color: #3fb950;
    font-size: 0.85rem;
    font-weight: 500;
    margin: 8px 0;
    letter-spacing: 0.02em;
}

.anomaly-tag {
    display: inline-block;
    background: #2d1b1b;
    border: 1px solid #f85149;
    color: #f85149;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 4px;
    margin: 3px 4px 3px 0;
}

.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7d8590;
    border-bottom: 1px solid #21262d;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 28px;
}

.credibility-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 10px;
}

.credibility-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.credibility-label {
    font-size: 0.8rem;
    color: #c9d1d9;
    font-family: 'DM Sans', sans-serif;
}

.credibility-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin: 4px 0 10px 0;
}

.verdict-excellent {
    display: inline-block;
    background: #0a1a0f;
    border: 1px solid #3fb950;
    color: #3fb950;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 4px;
}

.verdict-good {
    display: inline-block;
    background: #1a1200;
    border: 1px solid #d29922;
    color: #d29922;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 4px;
}

.verdict-poor {
    display: inline-block;
    background: #1a0a0a;
    border: 1px solid #f85149;
    color: #f85149;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 10px;
    border-radius: 4px;
}

.tag-normal {
    display: inline-block;
    background: #0a1a0f;
    border: 1px solid #3fb950;
    color: #3fb950;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
}

.model-badge {
    display: inline-block;
    background: #1a1f2e;
    border: 1px solid #1f6feb;
    color: #58a6ff;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 4px;
}

div[data-testid="stSlider"] label {
    color: #7d8590 !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
}

div[data-testid="stSelectbox"] label {
    color: #7d8590 !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
}

h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.4rem !important;
    color: #e6edf3 !important;
    letter-spacing: -0.01em !important;
}

h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    color: #c9d1d9 !important;
}

.stButton > button {
    background: #1f6feb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 8px 20px !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #388bfd !important;
}

.stSelectbox > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}

.stSlider > div {
    color: #e6edf3 !important;
}

hr {
    border-color: #21262d !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_train(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'air' in cl and 'temp' in cl:
            col_map[c] = 'air_temp'
        elif 'process' in cl and 'temp' in cl:
            col_map[c] = 'proc_temp'
        elif 'rotational' in cl or 'rpm' in cl:
            col_map[c] = 'rot_speed'
        elif 'torque' in cl:
            col_map[c] = 'torque'
        elif 'tool' in cl and 'wear' in cl:
            col_map[c] = 'tool_wear'
        elif 'type' in cl and len(c) < 10:
            col_map[c] = 'machine_type'
        elif 'machine failure' in cl or 'failure' in cl.replace(' ', ''):
            col_map[c] = 'failure'
    df = df.rename(columns=col_map)

    le = LabelEncoder()
    if 'machine_type' in df.columns:
        df['type_enc'] = le.fit_transform(df['machine_type'].astype(str))
    else:
        df['type_enc'] = 0

    max_wear = 250
    df['RUL'] = (max_wear - df['tool_wear']).clip(lower=0)

    features = ['air_temp', 'proc_temp', 'rot_speed', 'torque', 'tool_wear', 'type_enc']
    available = [f for f in features if f in df.columns]

    X = df[available]
    y_rul = df['RUL']
    y_fail = df['failure'] if 'failure' in df.columns else (df['tool_wear'] > 200).astype(int)

    X_train, X_test, yr_train, yr_test, yf_train, yf_test = train_test_split(
        X, y_rul, y_fail, test_size=0.2, random_state=42
    )

    rul_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    rul_model.fit(X_train, yr_train)

    fail_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    fail_model.fit(X_train, yf_train)

    rul_preds = rul_model.predict(X_test)
    fail_preds = fail_model.predict(X_test)
    fail_proba = fail_model.predict_proba(X_test)[:, 1]

    stats = {
        'rul_mae': mean_absolute_error(yr_test, rul_preds),
        'rul_r2': r2_score(yr_test, rul_preds),
        'fail_acc': accuracy_score(yf_test, fail_preds),
        'feature_importance': dict(zip(available, rul_model.feature_importances_)),
        'df_stats': {
            'air_temp': (df['air_temp'].mean(), df['air_temp'].std()),
            'proc_temp': (df['proc_temp'].mean(), df['proc_temp'].std()),
            'rot_speed': (df['rot_speed'].mean(), df['rot_speed'].std()),
            'torque': (df['torque'].mean(), df['torque'].std()),
            'tool_wear': (df['tool_wear'].mean(), df['tool_wear'].std()),
        },
        'confusion': confusion_matrix(yf_test, fail_preds).tolist(),
        'n_samples': len(df),
        'failure_rate': df['failure'].mean() if 'failure' in df.columns else 0,
        'df': df,
        'available': available,
    }

    return rul_model, fail_model, stats, available, le


def compute_health_score(air_temp, proc_temp, rot_speed, torque, tool_wear):
    max_wear = 250
    wear_health = max(0, (1 - tool_wear / max_wear)) * 100
    temp_diff = proc_temp - air_temp
    temp_health = max(0, (1 - max(0, temp_diff - 8.6) / 5)) * 100
    torque_health = max(0, (1 - max(0, abs(torque - 40) - 20) / 30)) * 100
    rpm_health = max(0, (1 - max(0, abs(rot_speed - 1500) - 400) / 600)) * 100
    health = (
        0.35 * wear_health +
        0.25 * temp_health +
        0.25 * torque_health +
        0.15 * rpm_health
    )
    return round(health, 1), {
        'Tool Wear': round(wear_health, 1),
        'Temperature': round(temp_health, 1),
        'Torque': round(torque_health, 1),
        'Rotational Speed': round(rpm_health, 1)
    }


def detect_anomalies(air_temp, proc_temp, rot_speed, torque, tool_wear, stats):
    anomalies = []
    checks = {
        'air_temp': air_temp,
        'proc_temp': proc_temp,
        'rot_speed': rot_speed,
        'torque': torque,
        'tool_wear': tool_wear,
    }
    labels = {
        'air_temp': 'Air Temperature',
        'proc_temp': 'Process Temperature',
        'rot_speed': 'Rotational Speed',
        'torque': 'Torque',
        'tool_wear': 'Tool Wear',
    }
    for key, val in checks.items():
        if key in stats['df_stats']:
            mu, sigma = stats['df_stats'][key]
            if sigma > 0:
                z = abs((val - mu) / sigma)
                if z > 2.5:
                    anomalies.append(f"{labels[key]}  (z = {z:.2f})")
    temp_diff = proc_temp - air_temp
    if temp_diff > 14:
        anomalies.append(f"Temperature Differential  (diff = {temp_diff:.1f} K)")
    return anomalies


def compute_rul_rate(tool_wear, wear_rate=2.0):
    failure_threshold = 250
    if wear_rate <= 0:
        return 999
    return round((failure_threshold - tool_wear) / wear_rate, 1)


def get_credibility_verdict(r2, mae, acc):
    score = 0
    if r2 >= 0.95:
        score += 3
    elif r2 >= 0.85:
        score += 2
    elif r2 >= 0.70:
        score += 1

    if acc >= 0.97:
        score += 3
    elif acc >= 0.90:
        score += 2
    elif acc >= 0.80:
        score += 1

    if score >= 5:
        return "HIGH CONFIDENCE", "excellent"
    elif score >= 3:
        return "MODERATE CONFIDENCE", "good"
    else:
        return "LOW CONFIDENCE", "poor"


csv_path = "dataset.csv"
if not os.path.exists(csv_path):
    csv_path = st.sidebar.file_uploader("Upload dataset.csv", type="csv")
    if csv_path is None:
        st.info("Place dataset.csv in the same directory as app.py, or upload it.")
        st.stop()

try:
    rul_model, fail_model, stats, available_features, le = load_and_train(csv_path)
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

df = stats['df']

verdict_label, verdict_level = get_credibility_verdict(stats['rul_r2'], stats['rul_mae'], stats['fail_acc'])

st.markdown(f"""
<div class="sticky-nav">
    <div style="display:flex;align-items:center;">
        <span class="nav-title">Motor RUL Predictor</span>
        <span class="nav-subtitle">3-Phase Induction Motor &nbsp;·&nbsp; Predictive Maintenance</span>
        <span class="nav-badge">Random Forest</span>
    </div>
    <div class="nav-right">
        <span class="nav-stat">R² {stats['rul_r2']:.4f}</span>
        <span class="nav-stat">Acc {stats['fail_acc']*100:.1f}%</span>
        <span class="nav-stat">{stats['n_samples']:,} samples</span>
        <span class="nav-stat" style="color:{'#3fb950' if verdict_level=='excellent' else '#d29922' if verdict_level=='good' else '#f85149'};">{verdict_label}</span>
    </div>
</div>
<div class="main-content-offset"></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-title">Sensor Input Panel</div>', unsafe_allow_html=True)
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"], index=1)
    st.markdown('<div class="metric-sub" style="margin:12px 0 4px 0;">Thermal Sensors</div>', unsafe_allow_html=True)
    air_temp = st.slider("Air Temperature (K)", 290.0, 315.0, 300.0, 0.1)
    proc_temp = st.slider("Process Temperature (K)", 300.0, 325.0, 310.0, 0.1)
    st.markdown('<div class="metric-sub" style="margin:12px 0 4px 0;">Mechanical Sensors</div>', unsafe_allow_html=True)
    rot_speed = st.slider("Rotational Speed (rpm)", 1000, 2500, 1500, 10)
    torque = st.slider("Torque (Nm)", 5.0, 80.0, 40.0, 0.5)
    tool_wear = st.slider("Tool Wear (min)", 0, 250, 60, 1)
    st.markdown('<div class="metric-sub" style="margin:12px 0 4px 0;">Degradation Rate</div>', unsafe_allow_html=True)
    wear_rate = st.slider("Wear Rate (min/cycle)", 0.5, 10.0, 2.0, 0.5)
    st.markdown("---")
    run = st.button("Run Prediction")

st.markdown("""
<script>
(function() {
    function addToggle() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (!sidebar) { setTimeout(addToggle, 500); return; }
        var existing = window.parent.document.getElementById('sidebar-reopen-btn');
        if (existing) return;
        var btn = window.parent.document.createElement('button');
        btn.id = 'sidebar-reopen-btn';
        btn.innerText = 'SENSORS';
        btn.style.cssText = 'position:fixed;left:0;top:50%;transform:translateY(-50%);z-index:9998;background:#161b22;border:1px solid #30363d;border-left:none;border-radius:0 6px 6px 0;color:#58a6ff;padding:10px 6px;cursor:pointer;writing-mode:vertical-rl;font-family:monospace;font-size:0.65rem;letter-spacing:0.1em;display:none;';
        btn.onclick = function() {
            var collapseBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
            if (collapseBtn) collapseBtn.click();
        };
        window.parent.document.body.appendChild(btn);
        var observer = new MutationObserver(function() {
            var collapsed = sidebar.getAttribute('aria-expanded') === 'false' || sidebar.style.width === '0px';
            btn.style.display = collapsed ? 'block' : 'none';
        });
        observer.observe(sidebar, { attributes: true, attributeFilter: ['aria-expanded', 'style'] });
    }
    addToggle();
})();
</script>
""", unsafe_allow_html=True)

type_map = {'L': 0, 'M': 1, 'H': 2}
type_enc = type_map.get(machine_type, 1)

input_data = {
    'air_temp': air_temp,
    'proc_temp': proc_temp,
    'rot_speed': rot_speed,
    'torque': torque,
    'tool_wear': tool_wear,
    'type_enc': type_enc
}
input_df = pd.DataFrame([{k: input_data[k] for k in available_features if k in input_data}])

rul_pred = float(rul_model.predict(input_df)[0])
fail_prob = float(fail_model.predict_proba(input_df)[0][1]) * 100
health_score, health_breakdown = compute_health_score(air_temp, proc_temp, rot_speed, torque, tool_wear)
anomalies = detect_anomalies(air_temp, proc_temp, rot_speed, torque, tool_wear, stats)
rul_trend = compute_rul_rate(tool_wear, wear_rate)

if health_score >= 70:
    health_color = "#3fb950"
    health_label = "HEALTHY"
elif health_score >= 40:
    health_color = "#d29922"
    health_label = "DEGRADING"
else:
    health_color = "#f85149"
    health_label = "CRITICAL"

if fail_prob < 25:
    fp_color = "#3fb950"
elif fail_prob < 60:
    fp_color = "#d29922"
else:
    fp_color = "#f85149"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Health Score</div>
        <div class="metric-value" style="color:{health_color};">{health_score}%</div>
        <div class="metric-sub">{health_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RUL (Model)</div>
        <div class="metric-value" style="color:#58a6ff;">{rul_pred:.0f} min</div>
        <div class="metric-sub">RF regression estimate</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RUL (Trend)</div>
        <div class="metric-value" style="color:#79c0ff;">{rul_trend:.0f} cycles</div>
        <div class="metric-sub">at {wear_rate} min/cycle rate</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Failure Probability</div>
        <div class="metric-value" style="color:{fp_color};">{fail_prob:.1f}%</div>
        <div class="metric-sub">RF classifier output</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

if anomalies:
    for a in anomalies:
        st.markdown(f'<div class="alert-critical">Anomaly Detected &mdash; {a}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-ok">All sensor readings within normal operating range</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">System Analysis</div>', unsafe_allow_html=True)

left, right = st.columns([1.1, 1])

with left:
    fig_health = go.Figure()
    categories = list(health_breakdown.keys())
    values = list(health_breakdown.values())
    colors_bar = ['#3fb950' if v >= 70 else '#d29922' if v >= 40 else '#f85149' for v in values]
    fig_health.add_trace(go.Bar(
        x=categories, y=values,
        marker_color=colors_bar, marker_line_width=0,
        text=[f"{v:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(family='DM Mono', size=11, color='#c9d1d9'),
    ))
    fig_health.update_layout(
        title=dict(text="Factor Health Breakdown", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=11, family='DM Sans', color='#c9d1d9'), showline=False),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10, family='DM Mono'), range=[0, 115]),
        margin=dict(l=10, r=10, t=40, b=10), height=280, showlegend=False,
    )
    st.plotly_chart(fig_health, use_container_width=True)

with right:
    fi = stats['feature_importance']
    fi_labels = {
        'air_temp': 'Air Temp', 'proc_temp': 'Proc Temp',
        'rot_speed': 'RPM', 'torque': 'Torque',
        'tool_wear': 'Tool Wear', 'type_enc': 'Machine Type'
    }
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    fi_names = [fi_labels.get(k, k) for k, v in fi_sorted]
    fi_vals = [v * 100 for k, v in fi_sorted]
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi_vals, y=fi_names, orientation='h',
        marker_color='#1f6feb', marker_line_width=0,
        text=[f"{v:.1f}%" for v in fi_vals],
        textposition='outside',
        textfont=dict(family='DM Mono', size=10, color='#c9d1d9'),
    ))
    fig_fi.update_layout(
        title=dict(text="Feature Importance (RUL Model)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10, family='DM Mono'), range=[0, max(fi_vals) * 1.25]),
        yaxis=dict(gridcolor='#0d1117', tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        margin=dict(l=10, r=40, t=40, b=10), height=280, showlegend=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown('<div class="section-title">Degradation Trajectory</div>', unsafe_allow_html=True)

wear_points = np.arange(0, 251, 5)
type_enc_arr = np.full(len(wear_points), type_enc)
air_arr = np.full(len(wear_points), air_temp)
proc_arr = np.full(len(wear_points), proc_temp)
rpm_arr = np.full(len(wear_points), rot_speed)
torq_arr = np.full(len(wear_points), torque)

traj_df = pd.DataFrame({
    'air_temp': air_arr, 'proc_temp': proc_arr,
    'rot_speed': rpm_arr, 'torque': torq_arr,
    'tool_wear': wear_points, 'type_enc': type_enc_arr,
})[available_features]

rul_trajectory = rul_model.predict(traj_df)
fp_trajectory = fail_model.predict_proba(traj_df)[:, 1] * 100

fig_traj = go.Figure()
fig_traj.add_trace(go.Scatter(
    x=wear_points, y=rul_trajectory, name='RUL (min)',
    line=dict(color='#58a6ff', width=2), mode='lines',
))
fig_traj.add_trace(go.Scatter(
    x=wear_points, y=fp_trajectory, name='Failure Probability (%)',
    line=dict(color='#f85149', width=2, dash='dash'), mode='lines', yaxis='y2',
))
fig_traj.add_vline(
    x=tool_wear, line_color='#d29922', line_width=1.5, line_dash='dot',
    annotation_text=f"  Current: {tool_wear} min",
    annotation_font=dict(color='#d29922', size=11, family='DM Mono'),
    annotation_position='top right'
)
fig_traj.update_layout(
    plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
    font=dict(color='#7d8590', family='DM Sans'),
    xaxis=dict(title=dict(text='Tool Wear (min)', font=dict(size=11, color='#7d8590')), gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
    yaxis=dict(title=dict(text='RUL (min)', font=dict(size=11, color='#58a6ff')), gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
    yaxis2=dict(title=dict(text='Failure Probability (%)', font=dict(size=11, color='#f85149')), overlaying='y', side='right', range=[0, 105], tickfont=dict(family='DM Mono', size=10), showgrid=False),
    legend=dict(font=dict(size=11, family='DM Sans', color='#c9d1d9'), bgcolor='#161b22', bordercolor='#21262d', borderwidth=1, x=0.01, y=0.99),
    margin=dict(l=10, r=60, t=20, b=10), height=300,
)
st.plotly_chart(fig_traj, use_container_width=True)


st.markdown('<div class="section-title">Exploratory Data Analysis</div>', unsafe_allow_html=True)

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Distributions", "Correlations", "Failure Patterns"])

with eda_tab1:
    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        fig_tw = px.histogram(df, x='tool_wear', nbins=40, color_discrete_sequence=['#1f6feb'])
        fig_tw.update_layout(
            title=dict(text="Tool Wear Distribution", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', title='Tool Wear (min)', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', title='Count', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False,
        )
        st.plotly_chart(fig_tw, use_container_width=True)

    with eda_c2:
        fig_torq = px.histogram(df, x='torque', nbins=40, color_discrete_sequence=['#3fb950'])
        fig_torq.update_layout(
            title=dict(text="Torque Distribution", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', title='Torque (Nm)', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', title='Count', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False,
        )
        st.plotly_chart(fig_torq, use_container_width=True)

    eda_c3, eda_c4 = st.columns(2)
    with eda_c3:
        fig_rpm = px.histogram(df, x='rot_speed', nbins=40, color_discrete_sequence=['#d29922'])
        fig_rpm.update_layout(
            title=dict(text="Rotational Speed Distribution", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', title='RPM', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', title='Count', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False,
        )
        st.plotly_chart(fig_rpm, use_container_width=True)

    with eda_c4:
        type_counts = df['machine_type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        fig_type = px.bar(type_counts, x='Type', y='Count', color_discrete_sequence=['#79c0ff'])
        fig_type.update_layout(
            title=dict(text="Machine Type Distribution", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', title='Machine Type', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', title='Count', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False,
        )
        st.plotly_chart(fig_type, use_container_width=True)

with eda_tab2:
    num_cols = ['air_temp', 'proc_temp', 'rot_speed', 'torque', 'tool_wear', 'RUL']
    corr_labels = ['Air Temp', 'Proc Temp', 'RPM', 'Torque', 'Tool Wear', 'RUL']
    corr_matrix = df[num_cols].corr().values

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_labels, y=corr_labels,
        colorscale=[[0, '#f85149'], [0.5, '#0d1117'], [1, '#1f6feb']],
        zmin=-1, zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, family='DM Mono', color='#e6edf3'),
        showscale=True,
    ))
    fig_corr.update_layout(
        title=dict(text="Feature Correlation Matrix", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        yaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        margin=dict(l=10, r=10, t=40, b=10), height=380,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    scatter_c1, scatter_c2 = st.columns(2)
    with scatter_c1:
        fig_sc1 = px.scatter(df.sample(min(2000, len(df))), x='tool_wear', y='torque',
                             color='RUL', color_continuous_scale=['#f85149', '#d29922', '#3fb950'],
                             opacity=0.5)
        fig_sc1.update_layout(
            title=dict(text="Tool Wear vs Torque (colored by RUL)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=300,
        )
        st.plotly_chart(fig_sc1, use_container_width=True)

    with scatter_c2:
        fig_sc2 = px.scatter(df.sample(min(2000, len(df))), x='rot_speed', y='torque',
                             color='RUL', color_continuous_scale=['#f85149', '#d29922', '#3fb950'],
                             opacity=0.5)
        fig_sc2.update_layout(
            title=dict(text="RPM vs Torque (colored by RUL)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=300,
        )
        st.plotly_chart(fig_sc2, use_container_width=True)

with eda_tab3:
    fail_col1, fail_col2 = st.columns(2)
    with fail_col1:
        fail_by_type = df.groupby('machine_type')['failure'].agg(['sum', 'count']).reset_index()
        fail_by_type.columns = ['Type', 'Failures', 'Total']
        fail_by_type['Rate'] = (fail_by_type['Failures'] / fail_by_type['Total'] * 100).round(2)
        fig_fbt = px.bar(fail_by_type, x='Type', y='Rate', color_discrete_sequence=['#f85149'],
                         text=[f"{r:.1f}%" for r in fail_by_type['Rate']])
        fig_fbt.update_layout(
            title=dict(text="Failure Rate by Machine Type (%)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=280, showlegend=False,
        )
        st.plotly_chart(fig_fbt, use_container_width=True)

    with fail_col2:
        wear_bins = pd.cut(df['tool_wear'], bins=[0, 50, 100, 150, 200, 250], labels=['0-50', '50-100', '100-150', '150-200', '200-250'])
        fail_by_wear = df.groupby(wear_bins, observed=True)['failure'].mean().reset_index()
        fail_by_wear.columns = ['Wear Range', 'Failure Rate']
        fail_by_wear['Failure Rate'] = (fail_by_wear['Failure Rate'] * 100).round(2)
        fig_fbw = px.bar(fail_by_wear, x='Wear Range', y='Failure Rate', color_discrete_sequence=['#d29922'],
                         text=[f"{r:.1f}%" for r in fail_by_wear['Failure Rate']])
        fig_fbw.update_layout(
            title=dict(text="Failure Rate by Tool Wear Band (%)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
            plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
            font=dict(color='#7d8590', family='DM Sans'),
            xaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            yaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
            margin=dict(l=10, r=10, t=40, b=10), height=280, showlegend=False,
        )
        st.plotly_chart(fig_fbw, use_container_width=True)

    rul_by_type = df.groupby('machine_type')['RUL'].mean().reset_index()
    rul_by_type.columns = ['Type', 'Avg RUL']
    fig_rbt = px.bar(rul_by_type, x='Type', y='Avg RUL', color_discrete_sequence=['#58a6ff'],
                     text=[f"{r:.0f} min" for r in rul_by_type['Avg RUL']])
    fig_rbt.update_layout(
        title=dict(text="Average RUL by Machine Type", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(family='DM Mono', size=10)),
        margin=dict(l=10, r=10, t=40, b=10), height=240, showlegend=False,
    )
    st.plotly_chart(fig_rbt, use_container_width=True)


st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RUL Model MAE</div>
        <div class="metric-value" style="font-size:1.5rem;color:#58a6ff;">{stats['rul_mae']:.2f}</div>
        <div class="metric-sub">minutes</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">RUL Model R²</div>
        <div class="metric-value" style="font-size:1.5rem;color:#58a6ff;">{stats['rul_r2']:.4f}</div>
        <div class="metric-sub">coefficient of determination</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Classifier Accuracy</div>
        <div class="metric-value" style="font-size:1.5rem;color:#3fb950;">{stats['fail_acc']*100:.2f}%</div>
        <div class="metric-sub">failure detection</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Training Samples</div>
        <div class="metric-value" style="font-size:1.5rem;color:#c9d1d9;">{stats['n_samples']:,}</div>
        <div class="metric-sub">80/20 train-test split</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="section-title">Model Credibility</div>', unsafe_allow_html=True)

r2_pct = min(100, stats['rul_r2'] * 100)
acc_pct = stats['fail_acc'] * 100
mae_score = max(0, 100 - stats['rul_mae'])

r2_verdict = "excellent" if stats['rul_r2'] >= 0.95 else "good" if stats['rul_r2'] >= 0.80 else "poor"
acc_verdict = "excellent" if acc_pct >= 97 else "good" if acc_pct >= 90 else "poor"
mae_verdict = "excellent" if stats['rul_mae'] <= 5 else "good" if stats['rul_mae'] <= 15 else "poor"

r2_color = "#3fb950" if r2_verdict == "excellent" else "#d29922" if r2_verdict == "good" else "#f85149"
acc_color = "#3fb950" if acc_verdict == "excellent" else "#d29922" if acc_verdict == "good" else "#f85149"
mae_color = "#3fb950" if mae_verdict == "excellent" else "#d29922" if mae_verdict == "good" else "#f85149"

cred_left, cred_right = st.columns([1.2, 1])

with cred_left:
    st.markdown(f"""
    <div class="credibility-card">
        <div style="margin-bottom:14px;">
            <div class="credibility-row">
                <span class="credibility-label">Regression Fit (R²) &mdash; how well the model explains RUL variance</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{r2_color};">{stats['rul_r2']:.4f}</span>
            </div>
            <div class="credibility-bar-bg">
                <div style="background:{r2_color};height:6px;border-radius:4px;width:{r2_pct:.1f}%;"></div>
            </div>
            <span class="{'verdict-excellent' if r2_verdict=='excellent' else 'verdict-good' if r2_verdict=='good' else 'verdict-poor'}">{'EXCELLENT — model captures nearly all variance' if r2_verdict=='excellent' else 'GOOD — most variance explained' if r2_verdict=='good' else 'NEEDS IMPROVEMENT'}</span>
        </div>
        <div style="margin-bottom:14px;">
            <div class="credibility-row">
                <span class="credibility-label">Classifier Accuracy &mdash; correct failure / non-failure predictions</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{acc_color};">{acc_pct:.2f}%</span>
            </div>
            <div class="credibility-bar-bg">
                <div style="background:{acc_color};height:6px;border-radius:4px;width:{acc_pct:.1f}%;"></div>
            </div>
            <span class="{'verdict-excellent' if acc_verdict=='excellent' else 'verdict-good' if acc_verdict=='good' else 'verdict-poor'}">{'EXCELLENT — near-perfect failure detection' if acc_verdict=='excellent' else 'GOOD — reliable detection' if acc_verdict=='good' else 'NEEDS IMPROVEMENT'}</span>
        </div>
        <div>
            <div class="credibility-row">
                <span class="credibility-label">Mean Absolute Error &mdash; average RUL prediction error</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.8rem;color:{mae_color};">{stats['rul_mae']:.2f} min</span>
            </div>
            <div class="credibility-bar-bg">
                <div style="background:{mae_color};height:6px;border-radius:4px;width:{min(100,mae_score):.1f}%;"></div>
            </div>
            <span class="{'verdict-excellent' if mae_verdict=='excellent' else 'verdict-good' if mae_verdict=='good' else 'verdict-poor'}">{'EXCELLENT — predictions within ±5 min' if mae_verdict=='excellent' else 'GOOD — predictions within ±15 min' if mae_verdict=='good' else 'HIGH ERROR — treat predictions with caution'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    overall_verdict_html = f'<span class="verdict-{verdict_level}">{verdict_label}</span>'
    st.markdown(f"""
    <div class="credibility-card" style="margin-top:10px;">
        <div class="metric-label" style="margin-bottom:8px;">Overall Prediction Credibility</div>
        <div style="display:flex;align-items:center;gap:12px;">
            {overall_verdict_html}
            <span class="metric-sub">Based on R², Accuracy, and MAE thresholds</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with cred_right:
    cm = np.array(stats['confusion'])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No Fail', 'Predicted: Fail'],
        y=['Actual: No Fail', 'Actual: Fail'],
        colorscale=[[0, '#0d1117'], [0.5, '#1f3a5f'], [1, '#1f6feb']],
        text=cm, texttemplate="%{text}",
        textfont=dict(size=16, color='white', family='DM Mono'),
        showscale=False,
    ))
    fig_cm.update_layout(
        title=dict(text="Confusion Matrix", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        yaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        margin=dict(l=10, r=10, t=40, b=10), height=260,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
n_samp = stats['n_samples']
st.markdown(f'<div class="metric-sub" style="text-align:center;padding:8px 0;">Ubiquitous Computing &nbsp;|&nbsp; 7-A &nbsp;|&nbsp; Bahria University &nbsp;|&nbsp; Dataset: {n_samp:,} samples</div>', unsafe_allow_html=True)