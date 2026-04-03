import streamlit as st
import requests
import pandas as pd
import time
import random
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

# ==========================================
# 1. CONFIGURATION
# ==========================================
API_URL = "http://127.0.0.1:8000/check_fraud"
st.set_page_config(page_title="UPI Sentinel Admin", layout="wide", page_icon="🛡️")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    div[data-testid="stMetricValue"] > div {
         color: #00FF00;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ UPI Sentinel: Hybrid Defense Console")
st.markdown("### 🧠 Supervised XGBoost + 👽 Unsupervised Isolation Forest + 🕸️ Interactive Graph ML")

# ==========================================
# 2. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🕹️ Simulation Control")
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔴 Start Live Traffic", value=False)
speed = st.sidebar.slider("Traffic Speed (sec)", 0.1, 2.0, 0.5)

st.sidebar.markdown("---")
# THE NEW GRAPH BUTTON
show_graph = st.sidebar.checkbox("🕸️ Enable Interactive Graph", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Logs"):
    st.session_state['history'] = []
    st.session_state['stats'] = {'total': 0, 'fraud': 0, 'anomaly': 0, 'blocked_amount': 0}
    st.rerun()

# ==========================================
# 3. STATE MANAGEMENT
# ==========================================
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'stats' not in st.session_state:
    st.session_state['stats'] = {'total': 0, 'fraud': 0, 'anomaly': 0, 'blocked_amount': 0}

# ==========================================
# 4. RANDOM TRAFFIC GENERATOR
# ==========================================
def generate_random_traffic():
    dice = random.random()
    if dice < 0.85:
        # Safe
        return {
            "amount": random.randint(50, 2000), "hour": random.randint(8, 22), "is_weekend": 0, "velocity_1h": random.randint(0, 3), "distance_from_home": random.uniform(0, 5), "is_new_device": 0, "is_new_recipient": 0, "account_age_days": random.randint(500, 2000)
        }
    elif dice < 0.93:
         # Known Fraud
         return {"amount": 50000, "hour": 3, "is_weekend": 1, "velocity_1h": 1, "distance_from_home": 10, "is_new_device": 0, "is_new_recipient": 0, "account_age_days": 1000}
    else:
        # Zero-Day Anomaly
        return {
            "amount": 9500, "hour": 23, "is_weekend": 1, "velocity_1h": 7, "distance_from_home": 150, "is_new_device": 1, "is_new_recipient": 1, "account_age_days": 2
        }

def format_fake_time(hour_int):
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"{hour_int:02d}:{minute:02d}:{second:02d}"

# ==========================================
# 5. INTERACTIVE GRAPH VISUALIZER (PyVis)
# ==========================================
def render_interactive_graph():
    """
    Creates a physics-enabled interactive network graph.
    """
    # 1. Initialize PyVis Network
    net = Network(height='500px', width='100%', bgcolor='#222222', font_color='white')
    
    # 2. Add The "Mule" (The Center of the Ring)
    net.add_node("MULE_ACC", label="🔴 MULE ACCOUNT", title="Risk Score: 99%", color='#FF0000', size=25)
    
    # 3. Add The "Boss" (Layering)
    net.add_node("BOSS_ACC", label="⚫ CRIME BOSS", title="Destination Wallet", color='#000000', size=35)
    net.add_edge("MULE_ACC", "BOSS_ACC", value=5, title="Transferred: ₹5,00,000", color='white')

    # 4. Add Victims (Star Pattern)
    for i in range(101, 115):
        user_id = f"User_{i}"
        amount = random.randint(5000, 25000)
        net.add_node(user_id, label=f"🟢 {user_id}", title=f"Safe User", color='#00FF00', size=15)
        # Connect to Mule
        net.add_edge(user_id, "MULE_ACC", value=2, title=f"Sent: ₹{amount}", color='gray')

    # 5. Physics Options (Make it bounce nicely)
    net.repulsion(node_distance=100, spring_length=200)

    # 6. Save and Render
    try:
        path = '/tmp'
        net.save_graph('fraud_network.html')
        HtmlFile = open('fraud_network.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=510)
    except:
        # Fallback for Windows local run if path issues
        net.save_graph('fraud_network.html')
        HtmlFile = open('fraud_network.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=510)
        
    st.caption("✨ **Interactive Mode:** Drag nodes to rearrange. Hover over lines to see transaction amounts.")

# ==========================================
# 6. MAIN DASHBOARD LAYOUT
# ==========================================
col1, col2, col3, col4 = st.columns(4)
metric_total = col1.empty()
metric_fraud = col2.empty()
metric_anomaly = col3.empty()
metric_saved = col4.empty()

st.divider()
alert_placeholder = st.empty()

# LOGIC: Show Interactive Graph IF button is clicked
if show_graph:
    st.subheader("🕸️ Interactive Graph ML: Mule Account Detection")
    render_interactive_graph() 
else:
    # NORMAL VIEW
    tab_all, tab_blocked, tab_approved = st.tabs(["📜 All Transactions", "🚨 Blocked Only", "✅ Approved Only"])
    with tab_all: table_all = st.empty()
    with tab_blocked: table_blocked = st.empty()
    with tab_approved: table_approved = st.empty()

    def render_tables():
        if len(st.session_state['history']) > 0:
            df = pd.DataFrame(st.session_state['history'])
            def highlight_rows(row):
                if "Zero-Day" in str(row['Type']): return ['background-color: #2e003e; color: #e0b0ff'] * len(row)
                elif "Fraud" in str(row['Type']): return ['background-color: #3d0000; color: #ffcccc'] * len(row)
                return [''] * len(row)
            
            table_all.dataframe(df.style.apply(highlight_rows, axis=1), use_container_width=True, height=400)
            
            df_blocked = df[df['Status'] == "🚨 BLOCKED"]
            if not df_blocked.empty: table_blocked.dataframe(df_blocked.style.apply(highlight_rows, axis=1), use_container_width=True, height=400)
            
            df_approved = df[df['Status'] == "✅ APPROVED"]
            if not df_approved.empty: table_approved.dataframe(df_approved, use_container_width=True, height=400)

# ==========================================
# 7. SIMULATION LOOP
# ==========================================
if auto_refresh:
    while True:
        txn = generate_random_traffic()
        try:
            response = requests.post(API_URL, json=txn)
            if response.status_code == 200:
                result = response.json()
                is_anomaly = result.get('anomaly_flag', 'No') == "Yes"
                st.session_state['stats']['total'] += 1
                
                if result['status'] == "🚨 BLOCKED":
                    st.session_state['stats']['blocked_amount'] += txn['amount']
                    if is_anomaly:
                        st.session_state['stats']['anomaly'] += 1
                        alert_placeholder.warning(f"⚠️ ANOMALY: {result['reason']}")
                        txn_type = "👽 Zero-Day"
                        display_score = "⚠️ Anomaly"
                    else:
                        st.session_state['stats']['fraud'] += 1
                        alert_placeholder.error(f"🛑 FRAUD: {result['reason']}")
                        txn_type = "🛑 Fraud"
                        display_score = result.get('fraud_probability', '0%')
                else:
                    alert_placeholder.success(f"✅ SECURE")
                    txn_type = "✅ Safe"
                    display_score = result.get('fraud_probability', '0%')

                new_row = {"Time": format_fake_time(txn['hour']), "Amount": f"₹{txn['amount']}", "Status": result['status'], "Reason": result['reason'], "Risk Score": display_score, "Type": txn_type, "Velocity": txn['velocity_1h'], "Distance": f"{int(txn['distance_from_home'])} km"}
                st.session_state['history'].insert(0, new_row)
                if len(st.session_state['history']) > 200: st.session_state['history'].pop()

                s = st.session_state['stats']
                metric_total.metric("Total Scanned", s['total'])
                metric_fraud.metric("Known Fraud", s['fraud']) 
                metric_anomaly.metric("Zero-Day Anomalies", s['anomaly'])
                metric_saved.metric("Money Saved", f"₹{s['blocked_amount']:,}")

                if not show_graph: # Only render tables if graph is hidden
                    render_tables()
            else:
                st.error("API Error")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            break
        time.sleep(speed)
else:
    s = st.session_state['stats']
    metric_total.metric("Total Scanned", s['total'])
    metric_fraud.metric("Known Fraud", s['fraud']) 
    metric_anomaly.metric("Zero-Day Anomalies", s['anomaly'])
    metric_saved.metric("Money Saved", f"₹{s['blocked_amount']:,}")
    if show_graph:
        st.subheader("🕸️ Interactive Graph ML: Mule Account Detection")
        render_interactive_graph()
    else:
        render_tables()
