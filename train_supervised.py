import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Generate Fake Training Data (30 features)
# 1000 samples, 30 columns
X = np.random.rand(1000, 30)
# Fake labels (0 = Safe, 1 = Fraud)
y = np.random.randint(0, 2, 1000)

# 2. Train a Simple Model (Random Forest is easier than XGBoost to install)
print("Training Supervised Model (P1)...")
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# 3. Save it as 'xgb_model.pkl' (Naming it this so your code finds it)
joblib.dump(model, "xgb_model.pkl")
print("SUCCESS: 'xgb_model.pkl' created! P1 should now be Online.")
