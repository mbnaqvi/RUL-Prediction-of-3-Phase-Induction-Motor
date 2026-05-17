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
import pickle

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

.health-bar-container {
    background: #21262d;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin: 8px 0;
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
        elif 'machine failure' in cl or 'failure' in cl.replace(' ',''):
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
    thresholds = {'z_score': 2.5}

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
                if z > thresholds['z_score']:
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


st.markdown("# Remaining Useful Life Prediction")
st.markdown('<div class="metric-sub" style="margin-bottom:24px;">3-Phase Induction Motor  |  Predictive Maintenance System  |  <span class="model-badge">Random Forest</span></div>', unsafe_allow_html=True)


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
        x=categories,
        y=values,
        marker_color=colors_bar,
        marker_line_width=0,
        text=[f"{v:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(family='DM Mono', size=11, color='#c9d1d9'),
    ))

    fig_health.update_layout(
        title=dict(text="Factor Health Breakdown", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=11, family='DM Sans', color='#c9d1d9'), showline=False),
        yaxis=dict(gridcolor='#21262d', tickfont=dict(size=10, family='DM Mono'), range=[0, 115]),
        margin=dict(l=10, r=10, t=40, b=10),
        height=280,
        showlegend=False,
    )
    st.plotly_chart(fig_health, use_container_width=True)

with right:
    fi = stats['feature_importance']
    fi_labels = {
        'air_temp': 'Air Temp',
        'proc_temp': 'Proc Temp',
        'rot_speed': 'RPM',
        'torque': 'Torque',
        'tool_wear': 'Tool Wear',
        'type_enc': 'Machine Type'
    }
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    fi_names = [fi_labels.get(k, k) for k, v in fi_sorted]
    fi_vals = [v * 100 for k, v in fi_sorted]

    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=fi_vals,
        y=fi_names,
        orientation='h',
        marker_color='#1f6feb',
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in fi_vals],
        textposition='outside',
        textfont=dict(family='DM Mono', size=10, color='#c9d1d9'),
    ))

    fig_fi.update_layout(
        title=dict(text="Feature Importance (RUL Model)", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#7d8590', family='DM Sans'),
        xaxis=dict(gridcolor='#21262d', tickfont=dict(size=10, family='DM Mono'), range=[0, max(fi_vals) * 1.25]),
        yaxis=dict(gridcolor='#0d1117', tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
        margin=dict(l=10, r=40, t=40, b=10),
        height=280,
        showlegend=False,
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
    'air_temp': air_arr,
    'proc_temp': proc_arr,
    'rot_speed': rpm_arr,
    'torque': torq_arr,
    'tool_wear': wear_points,
    'type_enc': type_enc_arr,
})[available_features]

rul_trajectory = rul_model.predict(traj_df)
fp_trajectory = fail_model.predict_proba(traj_df)[:, 1] * 100

fig_traj = go.Figure()

fig_traj.add_trace(go.Scatter(
    x=wear_points, y=rul_trajectory,
    name='RUL (min)',
    line=dict(color='#58a6ff', width=2),
    mode='lines',
))

fig_traj.add_trace(go.Scatter(
    x=wear_points, y=fp_trajectory,
    name='Failure Probability (%)',
    line=dict(color='#f85149', width=2, dash='dash'),
    mode='lines',
    yaxis='y2',
))

fig_traj.add_vline(
    x=tool_wear, line_color='#d29922', line_width=1.5, line_dash='dot',
    annotation_text=f"  Current: {tool_wear} min",
    annotation_font=dict(color='#d29922', size=11, family='DM Mono'),
    annotation_position='top right'
)

fig_traj.update_layout(
    plot_bgcolor='#0d1117',
    paper_bgcolor='#0d1117',
    font=dict(color='#7d8590', family='DM Sans'),
    xaxis=dict(
        title=dict(text='Tool Wear (min)', font=dict(size=11, color='#7d8590')),
        gridcolor='#21262d',
        tickfont=dict(family='DM Mono', size=10),
    ),
    yaxis=dict(
        title=dict(text='RUL (min)', font=dict(size=11, color='#58a6ff')),
        gridcolor='#21262d',
        tickfont=dict(family='DM Mono', size=10),
    ),
    yaxis2=dict(
        title=dict(text='Failure Probability (%)', font=dict(size=11, color='#f85149')),
        overlaying='y', side='right',
        range=[0, 105],
        tickfont=dict(family='DM Mono', size=10),
        showgrid=False,
    ),
    legend=dict(
        font=dict(size=11, family='DM Sans', color='#c9d1d9'),
        bgcolor='#161b22',
        bordercolor='#21262d',
        borderwidth=1,
        x=0.01, y=0.99,
    ),
    margin=dict(l=10, r=60, t=20, b=10),
    height=300,
)

st.plotly_chart(fig_traj, use_container_width=True)


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


cm = np.array(stats['confusion'])
fig_cm = go.Figure(data=go.Heatmap(
    z=cm,
    x=['Predicted: No Failure', 'Predicted: Failure'],
    y=['Actual: No Failure', 'Actual: Failure'],
    colorscale=[[0, '#0d1117'], [0.5, '#1f3a5f'], [1, '#1f6feb']],
    text=cm,
    texttemplate="%{text}",
    textfont=dict(size=16, color='white', family='DM Mono'),
    showscale=False,
))

fig_cm.update_layout(
    title=dict(text="Confusion Matrix", font=dict(size=13, color='#7d8590', family='DM Sans'), x=0),
    plot_bgcolor='#0d1117',
    paper_bgcolor='#0d1117',
    font=dict(color='#7d8590', family='DM Sans'),
    xaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
    yaxis=dict(tickfont=dict(size=11, family='DM Sans', color='#c9d1d9')),
    margin=dict(l=10, r=10, t=40, b=10),
    height=260,
)

col_cm, col_space = st.columns([1, 1])
with col_cm:
    st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")
n_samp = stats['n_samples']
st.markdown(f'<div class="metric-sub" style="text-align:center;padding:8px 0;">Ubiquitous Computing Assignment 3 &nbsp;|&nbsp; BS CS Semester 7 &nbsp;|&nbsp; Bahria University &nbsp;|&nbsp; Dataset: {n_samp:,} samples</div>', unsafe_allow_html=True)