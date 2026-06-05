import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, confusion_matrix
import pickle
import json
import warnings

warnings.filterwarnings("ignore")

CSV_PATH = "dataset.csv"
MODEL_OUT = "models/rul_model.pkl"
CLASSIFIER_OUT = "models/fail_model.pkl"
ENCODER_OUT = "models/label_encoder.pkl"
STATS_OUT = "models/model_stats.json"

import os
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
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

print(f"Dataset shape: {df.shape}")
print(f"Columns mapped: {list(col_map.values())}")

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

print(f"\nTraining samples: {len(X_train)}  |  Test samples: {len(X_test)}")

print("\nTraining RUL Regressor (Random Forest)...")
rul_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
rul_model.fit(X_train, yr_train)

print("Training Failure Classifier (Random Forest)...")
fail_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
fail_model.fit(X_train, yf_train)

rul_preds = rul_model.predict(X_test)
fail_preds = fail_model.predict(X_test)

rul_mae = mean_absolute_error(yr_test, rul_preds)
rul_r2 = r2_score(yr_test, rul_preds)
fail_acc = accuracy_score(yf_test, fail_preds)
cm = confusion_matrix(yf_test, fail_preds).tolist()

print(f"\n--- Results ---")
print(f"RUL MAE   : {rul_mae:.4f} min")
print(f"RUL R²    : {rul_r2:.4f}")
print(f"Fail Acc  : {fail_acc*100:.2f}%")
print(f"Confusion :\n{np.array(cm)}")

stats = {
    'rul_mae': rul_mae,
    'rul_r2': rul_r2,
    'fail_acc': fail_acc,
    'feature_importance': dict(zip(available, rul_model.feature_importances_.tolist())),
    'df_stats': {
        'air_temp': [float(df['air_temp'].mean()), float(df['air_temp'].std())],
        'proc_temp': [float(df['proc_temp'].mean()), float(df['proc_temp'].std())],
        'rot_speed': [float(df['rot_speed'].mean()), float(df['rot_speed'].std())],
        'torque': [float(df['torque'].mean()), float(df['torque'].std())],
        'tool_wear': [float(df['tool_wear'].mean()), float(df['tool_wear'].std())],
    },
    'confusion': cm,
    'n_samples': len(df),
    'failure_rate': float(df['failure'].mean()) if 'failure' in df.columns else 0.0,
    'available_features': available,
}

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(rul_model, f)

with open(CLASSIFIER_OUT, 'wb') as f:
    pickle.dump(fail_model, f)

with open(ENCODER_OUT, 'wb') as f:
    pickle.dump(le, f)

with open(STATS_OUT, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved: {MODEL_OUT}")
print(f"Saved: {CLASSIFIER_OUT}")
print(f"Saved: {ENCODER_OUT}")
print(f"Saved: {STATS_OUT}")