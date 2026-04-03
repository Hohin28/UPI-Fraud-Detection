import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 🛑 STEP 0: GENERATE DUMMY DATA (For Testing)
# ==========================================
# (If you have a real csv, replace this block with pd.read_csv)
def create_dummy_dataset(n=10000):
    print("⚠️ No dataset found. Generating dummy data for testing...")
    data = {
        'amount': np.random.exponential(scale=500, size=n), # Skewed amounts
        'hour': np.random.randint(0, 24, size=n),
        'channel': np.random.choice(['mobile_app', 'web', 'third_party', 'merchant_pos'], size=n),
        'is_fraud': np.random.choice([0, 1], size=n, p=[0.95, 0.05]) # 5% fraud rate
    }
    df = pd.DataFrame(data)
    # Inject a simple pattern for the model to find
    # (High amounts at 3 AM are likely fraud)
    df.loc[(df['hour'] == 3) & (df['amount'] > 1000), 'is_fraud'] = 1
    return df

# Try to load file, else create dummy
try:
    df = pd.read_csv('upi_transactions.csv')
    print("✅ Loaded 'upi_transactions.csv' successfully!")
except FileNotFoundError:
    df = create_dummy_dataset()

# ==========================================
# ✅ TASK 1: INSPECT DATASET
# ==========================================
print("\n" + "="*40)
print("📊 TASK 1: DATA INSPECTION")
print("="*40)

# 1. Total Rows
print(f"Total Transactions: {len(df)}")

# 2. % Fraud vs Legit
fraud_count = df['is_fraud'].sum()
legit_count = len(df) - fraud_count
fraud_pct = (fraud_count / len(df)) * 100
print(f"Fraud Breakdown: {fraud_count} Fraud ({fraud_pct:.2f}%) vs {legit_count} Legit")

# 3. Column Names
print(f"Columns: {list(df.columns)}")

# 4. Missing Values
print("\nMissing Values Check:")
print(df.isnull().sum())

# ==========================================
# ✅ TASK 2: BASELINE MODEL (Dumb Model)
# ==========================================
print("\n" + "="*40)
print("🤖 TASK 2: TRAINING BASELINE MODEL")
print("="*40)

# 1. Feature Selection (As requested: amount, hour, channel)
features = ['amount', 'hour', 'channel']
target = 'is_fraud'

X = df[features].copy()
y = df[target]

# 2. Preprocessing (Label Encode 'channel')
le = LabelEncoder()
X['channel'] = le.fit_transform(X['channel'])
print(f"Encoded 'channel' mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Model (Random Forest)
print("\nTraining RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Predictions
y_pred = rf_model.predict(X_test)

# ==========================================
# 📈 RESULTS
# ==========================================
print("\n" + "="*40)
print("🏆 BASELINE RESULTS")
print("="*40)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.4f} (Don't trust this if classes are imbalanced!)")

# Recall (Crucial for Fraud)
rec = recall_score(y_test, y_pred)
print(f"🔥 Recall (Fraud Catch Rate): {rec:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n❌ Confusion Matrix:")
print(cm)
print("( [True Neg, False Pos] )")
print("( [False Neg, True Pos] )")

# Full Report
print("\n📄 Detailed Report:")
print(classification_report(y_test, y_pred))