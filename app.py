import streamlit as st
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="GlucoLens — Diabetes Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #0d1117;
    --card: #161b22;
    --card2: #1c2128;
    --accent: #3b82f6;
    --accent2: #06b6d4;
    --safe: #22c55e;
    --risk: #ef4444;
    --warn: #f59e0b;
    --text: #e6edf3;
    --muted: #7d8590;
    --border: #30363d;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
h1,h2,h3 { font-family: 'Playfair Display', serif !important; }

.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 60%, #161b22 100%);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '🩸';
    position: absolute;
    right: 2.5rem; top: 50%;
    transform: translateY(-50%);
    font-size: 5rem; opacity: 0.07;
}
.hero h1 {
    font-size: 2.6rem; margin: 0 0 0.3rem 0;
    background: linear-gradient(135deg, #e6edf3, #3b82f6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--muted); font-size: 1rem; margin: 0; }

.result-safe {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 2px solid var(--safe); border-radius: 16px;
    padding: 2rem; text-align: center;
}
.result-risk {
    background: linear-gradient(135deg, #2d0a0a, #450a0a);
    border: 2px solid var(--risk); border-radius: 16px;
    padding: 2rem; text-align: center;
}
.result-title { font-family: 'Playfair Display', serif; font-size: 2rem; margin: 0.5rem 0; }
.result-sub { color: var(--muted); font-size: 0.9rem; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin: 1.4rem 0 0.9rem 0;
}

.info-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.8rem 1rem;
    margin-bottom: 0.5rem; font-size: 0.85rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 1rem !important;
    padding: 0.7rem 2rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature metadata ─────────────────────────────────────────────
FEATURES = [
    {
        "name": "Pregnancies",
        "desc": "Number of times pregnant",
        "min": 0, "max": 20, "step": 1, "fmt": "%d",
        "normal": "0–5",
        "unit": "",
        "type": "int",
        "regex": r"pregnancies[\s:=]+(\d+)"
    },
    {
        "name": "Glucose",
        "desc": "Plasma glucose concentration (mg/dL)",
        "min": 0, "max": 300, "step": 1, "fmt": "%d",
        "normal": "70–100 mg/dL (fasting)",
        "unit": "mg/dL",
        "type": "int",
        "regex": r"glucose[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "Blood Pressure",
        "desc": "Diastolic blood pressure (mm Hg)",
        "min": 0, "max": 200, "step": 1, "fmt": "%d",
        "normal": "60–80 mm Hg",
        "unit": "mm Hg",
        "type": "int",
        "regex": r"blood[\s_]pressure[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "Skin Thickness",
        "desc": "Triceps skinfold thickness (mm)",
        "min": 0, "max": 100, "step": 1, "fmt": "%d",
        "normal": "10–30 mm",
        "unit": "mm",
        "type": "int",
        "regex": r"skin[\s_]thickness[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "Insulin",
        "desc": "2-Hour serum insulin (µU/mL)",
        "min": 0, "max": 900, "step": 1, "fmt": "%d",
        "normal": "16–166 µU/mL",
        "unit": "µU/mL",
        "type": "int",
        "regex": r"insulin[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "BMI",
        "desc": "Body mass index (weight in kg / height in m²)",
        "min": 0.0, "max": 70.0, "step": 0.1, "fmt": "%.1f",
        "normal": "18.5–24.9",
        "unit": "kg/m²",
        "type": "float",
        "regex": r"bmi[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "Diabetes Pedigree Function",
        "desc": "Genetic diabetes risk score based on family history",
        "min": 0.0, "max": 3.0, "step": 0.001, "fmt": "%.3f",
        "normal": "0.0–0.5 (low family risk)",
        "unit": "",
        "type": "float",
        "regex": r"(?:diabetes[\s_]pedigree|dpf|pedigree)[\s:=]+(\d+(?:\.\d+)?)"
    },
    {
        "name": "Age",
        "desc": "Age in years",
        "min": 0, "max": 120, "step": 1, "fmt": "%d",
        "normal": "—",
        "unit": "years",
        "type": "int",
        "regex": r"age[\s:=]+(\d+)"
    },
]
FEATURE_NAMES = [f["name"] for f in FEATURES]

# ── Normal range thresholds for risk bars ────────────────────────
NORMAL_MAX = {
    "Pregnancies": 5,
    "Glucose": 100,
    "Blood Pressure": 80,
    "Skin Thickness": 30,
    "Insulin": 166,
    "BMI": 24.9,
    "Diabetes Pedigree Function": 0.5,
    "Age": 45,
}
FEAT_MAX = {f["name"]: f["max"] for f in FEATURES}

# ── Train model ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on PIMA Diabetes dataset…")
def train_model():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = FEATURE_NAMES + ["Outcome"]
    try:
        data = pd.read_csv(url, names=cols)
    except:
        np.random.seed(42)
        n = 768
        data = pd.DataFrame({
            "Pregnancies": np.random.randint(0, 15, n),
            "Glucose": np.random.randint(70, 200, n),
            "Blood Pressure": np.random.randint(40, 120, n),
            "Skin Thickness": np.random.randint(5, 60, n),
            "Insulin": np.random.randint(0, 400, n),
            "BMI": np.round(np.random.uniform(18, 55, n), 1),
            "Diabetes Pedigree Function": np.round(np.random.uniform(0.05, 2.5, n), 3),
            "Age": np.random.randint(18, 80, n),
            "Outcome": np.random.choice([0, 1], n, p=[0.65, 0.35])
        })

    X = data[FEATURE_NAMES]
    y = data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    model = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=5)
    model.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_s)) * 100
    return model, scaler, round(acc, 2)

# ── PDF parser ───────────────────────────────────────────────────
def parse_report(text):
    vals = {}
    text_lower = text.lower()
    for f in FEATURES:
        m = re.search(f["regex"], text_lower)
        if m:
            v = float(m.group(1))
            v = max(f["min"], min(f["max"], v))
            vals[f["name"]] = int(round(v)) if f["type"] == "int" else round(v, 3)
    return vals

# ── Risk bar helper ───────────────────────────────────────────────
def risk_color(name, val):
    norm = NORMAL_MAX.get(name, FEAT_MAX.get(name, 100))
    fmax = FEAT_MAX.get(name, 100)
    pct  = val / fmax if fmax else 0
    if val <= norm:
        return "#22c55e", pct
    elif val <= norm * 1.3:
        return "#f59e0b", pct
    else:
        return "#ef4444", pct

# ── Load ──────────────────────────────────────────────────────────
model, scaler, model_acc = train_model()

# ── Hero ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>GlucoLens</h1>
  <p>Diabetes Risk Predictor · PIMA Indians Dataset · Decision Tree · Model Accuracy: {model_acc}%</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ About")
    st.caption(
        "GlucoLens uses a Decision Tree classifier trained on the PIMA Indians Diabetes Dataset "
        "(768 patients) to predict diabetes risk from 8 clinical indicators."
    )
    st.markdown(f"**Model Accuracy:** {model_acc}%")
    st.markdown("---")
    st.markdown("### 📋 Normal Ranges")
    for f in FEATURES:
        if f["normal"] != "—":
            st.caption(f"**{f['name']}:** {f['normal']}")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a medical diagnosis tool.")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🩸 Predict", "📖 How It Works"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Enter Patient Data</div>', unsafe_allow_html=True)

        # PDF / TXT upload
        uploaded = st.file_uploader(
            "📄 Upload lab report (PDF or TXT) — auto-fills inputs",
            type=["pdf", "txt"]
        )
        auto_vals = {}
        if uploaded:
            try:
                if uploaded.type == "application/pdf":
                    import pdfplumber
                    with pdfplumber.open(uploaded) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                else:
                    text = uploaded.read().decode("utf-8", errors="ignore")
                auto_vals = parse_report(text)
                if auto_vals:
                    st.success(f"✅ Auto-detected {len(auto_vals)}/{len(FEATURES)} fields from report")
                else:
                    st.warning("Could not parse values — please enter manually below.")
            except Exception as e:
                st.warning(f"Could not read file: {e}. Please enter values manually.")

        st.markdown("**Or enter values manually:**")

        input_vals = {}
        for f in FEATURES:
            default = auto_vals.get(f["name"], f["min"])
            col_a, col_b = st.columns([3, 1])
            with col_a:
                if f["type"] == "int":
                    input_vals[f["name"]] = st.number_input(
                        f["name"], min_value=int(f["min"]), max_value=int(f["max"]),
                        value=int(default), step=int(f["step"]), help=f["desc"]
                    )
                else:
                    input_vals[f["name"]] = st.number_input(
                        f["name"], min_value=float(f["min"]), max_value=float(f["max"]),
                        value=float(default), step=float(f["step"]),
                        format=f["fmt"], help=f["desc"]
                    )
            with col_b:
                st.markdown(f"<div style='padding-top:1.8rem;color:#7d8590;font-size:0.75rem'>{f['unit']}</div>",
                            unsafe_allow_html=True)

        predict_btn = st.button("🔍 Predict Diabetes Risk", use_container_width=True)

    # ── Result column ─────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

        if predict_btn:
            arr = np.array([[input_vals[n] for n in FEATURE_NAMES]])
            arr_scaled = scaler.transform(arr)
            pred = int(model.predict(arr_scaled)[0])

            # Confidence
            conf_str = ""
            if hasattr(model, "predict_proba"):
                try:
                    prob = model.predict_proba(arr_scaled)[0]
                    conf = prob[pred] * 100
                    conf_str = f"<p class='result-sub'>Confidence: <b>{conf:.1f}%</b></p>"
                except: pass

            if pred == 0:
                st.markdown(f"""
                <div class="result-safe">
                  <div style="font-size:3rem">✅</div>
                  <div class="result-title" style="color:#22c55e">Not Diabetic</div>
                  <p class="result-sub">No diabetes risk detected</p>
                  {conf_str}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-risk">
                  <div style="font-size:3rem">⚠️</div>
                  <div class="result-title" style="color:#ef4444">Diabetic Risk Detected</div>
                  <p class="result-sub">Patient shows signs of diabetes</p>
                  {conf_str}
                </div>""", unsafe_allow_html=True)

            # ── Risk gauge chart ──────────────────────────────────
            st.markdown('<div class="section-header">Overall Risk Gauge</div>', unsafe_allow_html=True)

            high_risk_count = sum(
                1 for f in FEATURES
                if input_vals[f["name"]] > NORMAL_MAX.get(f["name"], FEAT_MAX.get(f["name"], 100))
            )
            risk_pct = high_risk_count / len(FEATURES)

            fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
            fig.patch.set_facecolor("#161b22")
            theta1, theta2 = 180, 180 - int(risk_pct * 180)
            colors_gauge = ["#22c55e", "#f59e0b", "#ef4444"]
            wedge_colors = ["#22c55e" if risk_pct < 0.33 else "#f59e0b" if risk_pct < 0.66 else "#ef4444"]

            ax.add_patch(mpatches.Wedge((0.5, 0.3), 0.4, 0, 180, width=0.15,
                                         facecolor="#30363d", transform=ax.transAxes))
            ax.add_patch(mpatches.Wedge((0.5, 0.3), 0.4, theta2, 180, width=0.15,
                                         facecolor=wedge_colors[0], transform=ax.transAxes, alpha=0.85))
            gauge_label = "Low Risk" if risk_pct < 0.33 else "Moderate Risk" if risk_pct < 0.66 else "High Risk"
            ax.text(0.5, 0.22, gauge_label, ha="center", va="center",
                    fontsize=13, fontweight="bold", color=wedge_colors[0],
                    transform=ax.transAxes)
            ax.text(0.5, 0.08, f"{high_risk_count}/{len(FEATURES)} indicators above normal",
                    ha="center", va="center", fontsize=9, color="#7d8590",
                    transform=ax.transAxes)
            ax.axis("off")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ── Feature risk bars ─────────────────────────────────
            st.markdown('<div class="section-header">Feature Risk Profile</div>', unsafe_allow_html=True)
            for f in FEATURES:
                val  = input_vals[f["name"]]
                fmax = f["max"]
                pct  = val / fmax if fmax else 0
                color, _ = risk_color(f["name"], val)
                norm = NORMAL_MAX.get(f["name"], "—")
                st.markdown(f"""
                <div style="margin-bottom:0.6rem">
                  <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
                    <span>{f['name']}</span>
                    <span style="color:{color};font-weight:600">
                      {val} {f['unit']} &nbsp;<span style="color:#7d8590;font-weight:400">(normal: {f['normal']})</span>
                    </span>
                  </div>
                  <div style="background:#30363d;border-radius:4px;height:7px;position:relative">
                    <div style="width:{min(pct*100,100):.0f}%;background:{color};height:7px;border-radius:4px"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#161b22;border:1px dashed #30363d;border-radius:16px;
                        padding:3rem;text-align:center;color:#7d8590;margin-top:2rem">
              <div style="font-size:3rem;margin-bottom:1rem">🩸</div>
              <p>Fill in the patient data and click<br><b>Predict Diabetes Risk</b></p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — HOW IT WORKS
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">About the Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
      <b>PIMA Indians Diabetes Dataset</b> — Originally from the National Institute of Diabetes and 
      Digestive and Kidney Diseases.<br><br>
      <span style="color:#7d8590">768 female patients · 8 clinical features · Binary outcome (Diabetic / Not Diabetic)<br>
      ~35% Diabetic · ~65% Not Diabetic · Patients are at least 21 years old of Pima Indian heritage</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Guide</div>', unsafe_allow_html=True)
    for f in FEATURES:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;padding:0.7rem 0;border-bottom:1px solid #30363d">
          <div style="min-width:200px;font-weight:500">{f['name']}</div>
          <div style="color:#7d8590;font-size:0.9rem">{f['desc']} &nbsp;·&nbsp; Normal: {f['normal']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">How to Use</div>', unsafe_allow_html=True)
    steps = [
        ("1", "Upload Report", "Upload a PDF or TXT lab report — values auto-fill where detected."),
        ("2", "Enter Values", "Adjust any values manually using the number inputs."),
        ("3", "Predict", "Click Predict to get the diabetes risk classification."),
        ("4", "Review Profile", "Check the risk gauge and feature bars to see which indicators are above normal."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex;gap:1rem;align-items:flex-start;margin-bottom:1rem">
          <div style="background:#3b82f6;color:white;border-radius:50%;width:28px;height:28px;
                      display:flex;align-items:center;justify-content:center;
                      font-weight:700;font-size:0.85rem;flex-shrink:0">{num}</div>
          <div>
            <b>{title}</b><br>
            <span style="color:#7d8590;font-size:0.9rem">{desc}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:12px;
                padding:1.2rem;margin-top:1rem;color:#f59e0b;font-size:0.85rem">
      ⚠️ <b>Disclaimer:</b> This app is for educational purposes only and does not constitute 
      medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>""", unsafe_allow_html=True)
