import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score

# ==========================================
# 1. GENERATE HYBRID DATASET
# ==========================================
def create_dataset(n=50000):
    np.random.seed(42)
    data = {
        'amount': np.random.exponential(scale=500, size=n),
        'hour': np.random.randint(0, 24, size=n),
        'is_weekend': np.random.choice([0, 1], size=n),
        'velocity_1h': np.random.choice(list(range(0, 60)), size=n),
        'distance_from_home': np.random.exponential(scale=5, size=n),
        'is_new_device': np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        'is_new_recipient': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        'account_age_days': np.random.randint(1, 3000, size=n),
        'is_fraud': np.zeros(n)
    }
    df = pd.DataFrame(data)

    # Inject Known Fraud Patterns (Supervised Training)
    df.loc[df['velocity_1h'] > 8, 'is_fraud'] = 1
    df.loc[(df['distance_from_home'] > 300) & (df['is_new_device'] == 1), 'is_fraud'] = 1
    df.loc[(df['hour'] <= 4) & (df['amount'] > 5000), 'is_fraud'] = 1
    df.loc[(df['is_new_recipient'] == 1) & (df['amount'] > 10000), 'is_fraud'] = 1

    return df

print("🛠️ Generating Data...")
df = create_dataset()

features = ['amount', 'hour', 'is_weekend', 'velocity_1h', 'distance_from_home', 
            'is_new_device', 'is_new_recipient', 'account_age_days']

X = df[features]
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. TRAIN BRAIN 1: XGBoost (Known Fraud)
# ==========================================
print("🚀 Training XGBoost (The Guard Dog)...")
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=10)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "fraud_model_brain.pkl")

# ==========================================
# 3. TRAIN BRAIN 2: Isolation Forest (Anomalies)
# ==========================================
print("👽 Training Isolation Forest (The Motion Sensor)...")
# Train ONLY on normal data so it learns what "Normal" looks like
X_normal = X_train[y_train == 0] 
iso_model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_model.fit(X_normal)
joblib.dump(iso_model, "anomaly_model.pkl")

print("✅ SUCCESS: Both AI Models Saved!")
