# 💳 UPI Fraud Detection System

A machine learning-based system designed to detect fraudulent UPI (Unified Payments Interface) transactions using anomaly detection and supervised learning techniques.

---

## 🧠 Overview

With the rapid growth of digital payments in India, UPI fraud has become a major concern.
This project aims to identify suspicious transactions by analyzing patterns and detecting anomalies in user behavior.

---

## 🚀 Features

* 🔍 Anomaly detection using unsupervised learning
* 🤖 Supervised ML model for fraud classification
* 📊 Dashboard for monitoring transactions
* 🌐 API server for real-time prediction
* 📉 Detection of unusual transaction patterns

---

## 🧩 Project Structure

* `api_server.py` → Backend API for fraud detection
* `dashboard.py` → Visualization dashboard
* `baseline_model.py` → Initial ML model
* `improved_model.py` → Optimized model
* `train_anomaly.py` → Anomaly detection training
* `train_supervised.py` → Supervised model training

### Models:

* `anomaly_model.pkl` → Trained anomaly detection model
* `fraud_model_brain.pkl` → Main classification model
* `xgb_model.pkl` → XGBoost model
* `anomaly_weights.pth` → Neural network weights

### Other:

* `fraud_network.html` → Visualization UI

---

## ⚙️ How It Works

1. Transaction data is collected

2. Features like:

   * transaction amount
   * time
   * frequency
   * user behavior
     are extracted

3. Two approaches are used:

   * **Anomaly Detection** → finds unusual behavior
   * **Supervised Learning** → classifies fraud vs normal

4. Final prediction is generated via API

---

## 🧪 Technologies Used

* Python 🐍
* Scikit-learn
* XGBoost
* PyTorch
* Flask / FastAPI (API server)
* HTML (Visualization)

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run API server

```bash
python api_server.py
```

### 3. Run dashboard

```bash
python dashboard.py
```

---

## 📊 Use Cases

* Banking fraud detection
* UPI transaction monitoring
* Fintech security systems
* Real-time fraud prevention

---

## ⚠️ Limitations

* Model accuracy depends on dataset quality
* False positives may occur
* Needs real-time data integration for production use

---

## 🔮 Future Improvements

* Real-time streaming detection
* Deep learning enhancements
* Integration with banking APIs
* Mobile app interface

---

## 👨‍💻 Author

Hohin – B.Tech Cybersecurity Student

---

## 💬 Summary

This project demonstrates how machine learning can be applied to detect financial fraud in real-world systems, combining cybersecurity and AI techniques.
