from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load BOTH Brains
try:
    xgb_brain = joblib.load('fraud_model_brain.pkl')
    iso_brain = joblib.load('anomaly_model.pkl')
    print("✅ Models loaded successfully")
except:
    print("❌ ERROR: Run improved_model.py first!")

app = FastAPI()

class Transaction(BaseModel):
    amount: float
    hour: int
    is_weekend: int
    velocity_1h: int
    distance_from_home: float
    is_new_device: int
    is_new_recipient: int
    account_age_days: int

@app.post("/check_fraud")
def check_transaction(txn: Transaction):
    features = pd.DataFrame([txn.dict()])
    
    # 1. CHECK XGBoost (Known Fraud)
    prob = xgb_brain.predict_proba(features)[0][1]
    is_known_fraud = 1 if prob > 0.30 else 0
    
    # 2. CHECK Isolation Forest (Zero-Day Anomaly)
    # Output: -1 (Anomaly), 1 (Normal)
    iso_pred = iso_brain.predict(features)[0] 
    is_anomaly = 1 if iso_pred == -1 else 0

    reasons = []
    
    # Explainability Logic
    if is_known_fraud:
        if txn.velocity_1h > 8: reasons.append("High Velocity Bot")
        elif txn.distance_from_home > 300: reasons.append("Impossible Travel")
        elif txn.amount > 5000 and txn.hour <= 4: reasons.append("Night Heist Pattern")
        elif txn.is_new_recipient == 1 and txn.amount > 10000: reasons.append("New Recipient Scam")
        else: reasons.append("Complex Fraud Pattern")
        
    if is_anomaly and not is_known_fraud:
        reasons.append("Zero-Day Anomaly Detected")

    # FINAL DECISION: Block if EITHER model flags it
    if is_known_fraud or is_anomaly:
        status = "🚨 BLOCKED"
    else:
        status = "✅ APPROVED"

    return {
        "status": status,
        "fraud_probability": f"{prob:.2%}",
        "reason": ", ".join(reasons) if reasons else "Safe",
        "anomaly_flag": "Yes" if is_anomaly else "No"
    }
