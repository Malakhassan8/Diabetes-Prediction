import streamlit as st
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    --accent: #3b82f6;
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

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .val { font-size: 1.8rem; font-weight: 600; font-family: 'Playfair Display', serif; }
.metric-card .lbl { color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem; color: var(--text);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem; margin: 1.4rem 0 0.9rem 0;
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
    {"name": "Pregnancies",               "desc": "Number of times pregnant",                             "min": 0,   "max": 20,  "step": 1,     "fmt": "%d",   "normal": "0–5",                   "unit": "",      "type": "int",   "regex": r"pregnancies[\s:=]+(\d+)"},
    {"name": "Glucose",                   "desc": "Plasma glucose concentration (mg/dL)",                "min": 0,   "max": 300, "step": 1,     "fmt": "%d",   "normal": "70–100 mg/dL (fasting)", "unit": "mg/dL", "type": "int",   "regex": r"glucose[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "Blood Pressure",            "desc": "Diastolic blood pressure (mm Hg)",                    "min": 0,   "max": 200, "step": 1,     "fmt": "%d",   "normal": "60–80 mm Hg",            "unit": "mm Hg", "type": "int",   "regex": r"blood[\s_]pressure[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "Skin Thickness",            "desc": "Triceps skinfold thickness (mm)",                     "min": 0,   "max": 100, "step": 1,     "fmt": "%d",   "normal": "10–30 mm",               "unit": "mm",    "type": "int",   "regex": r"skin[\s_]thickness[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "Insulin",                   "desc": "2-Hour serum insulin (µU/mL)",                        "min": 0,   "max": 900, "step": 1,     "fmt": "%d",   "normal": "16–166 µU/mL",           "unit": "µU/mL", "type": "int",   "regex": r"insulin[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "BMI",                       "desc": "Body mass index (weight in kg / height in m²)",       "min": 0.0, "max": 70.0,"step": 0.1,   "fmt": "%.1f", "normal": "18.5–24.9",              "unit": "kg/m²", "type": "float", "regex": r"bmi[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "Diabetes Pedigree Function","desc": "Genetic diabetes risk score based on family history", "min": 0.0, "max": 3.0, "step": 0.001, "fmt": "%.3f", "normal": "0.0–0.5 (low family risk)","unit": "",     "type": "float", "regex": r"(?:diabetes[\s_]pedigree|dpf|pedigree)[\s:=]+(\d+(?:\.\d+)?)"},
    {"name": "Age",                       "desc": "Age in years",                                        "min": 0,   "max": 120, "step": 1,     "fmt": "%d",   "normal": "—",                      "unit": "years", "type": "int",   "regex": r"age[\s:=]+(\d+)"},
]
FEATURE_NAMES = [f["name"] for f in FEATURES]

NORMAL_MAX = {
    "Pregnancies": 5, "Glucose": 100, "Blood Pressure": 80,
    "Skin Thickness": 30, "Insulin": 166, "BMI": 24.9,
    "Diabetes Pedigree Function": 0.5, "Age": 45,
}
FEAT_MAX = {f["name"]: f["max"] for f in FEATURES}

# ── Train models at startup (no .sav files needed) ───────────────
@st.cache_resource(show_spinner="Training models… this takes ~10 seconds ⏳")
def load_models():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report

    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

    x = df.drop('Outcome', axis=1)
    y = df['Outcome']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled  = scaler.transform(x_test)

model_list = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM":                 SVC(random_state=42, probability=True),
    "KNN":                 KNeighborsClassifier(),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=50, random_state=42),
}

    models  = {}
    metrics = {}
    for name, m in model_list.items():
        m.fit(x_train_scaled, y_train)
        y_pred = m.predict(x_test_scaled)
        report = classification_report(y_test, y_pred, output_dict=True)
        models[name] = m
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "Precision": round(report['weighted avg']['precision'] * 100, 2),
            "Recall":    round(report['weighted avg']['recall'] * 100, 2),
            "F1-Score":  round(report['weighted avg']['f1-score'] * 100, 2),
        }

    return models, scaler, metrics

# ── Helpers ──────────────────────────────────────────────────────
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

def risk_color(name, val):
    norm = NORMAL_MAX.get(name, FEAT_MAX.get(name, 100))
    fmax = FEAT_MAX.get(name, 100)
    pct  = val / fmax if fmax else 0
    if val <= norm:         return "#22c55e", pct
    elif val <= norm * 1.3: return "#f59e0b", pct
    else:                   return "#ef4444", pct

def show_risk_gauge(input_vals):
    high_risk_count = sum(
        1 for f in FEATURES
        if input_vals[f["name"]] > NORMAL_MAX.get(f["name"], FEAT_MAX.get(f["name"], 100))
    )
    risk_pct  = high_risk_count / len(FEATURES)
    color     = "#22c55e" if risk_pct < 0.33 else "#f59e0b" if risk_pct < 0.66 else "#ef4444"
    label     = "Low Risk" if risk_pct < 0.33 else "Moderate Risk" if risk_pct < 0.66 else "High Risk"
    bar_width = int(risk_pct * 100)
    st.markdown('<div class="section-header">Overall Risk Gauge</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:1.2rem 1.5rem">
      <div style="display:flex;justify-content:space-between;margin-bottom:0.5rem;font-size:0.85rem">
        <span style="color:{color};font-weight:700">{label}</span>
        <span style="color:#7d8590">{high_risk_count}/{len(FEATURES)} indicators above normal</span>
      </div>
      <div style="background:#30363d;border-radius:6px;height:14px">
        <div style="width:{bar_width}%;background:{color};height:14px;border-radius:6px"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def show_risk_bars(input_vals):
    st.markdown('<div class="section-header" style="margin-top:1.2rem">Feature Risk Profile</div>', unsafe_allow_html=True)
    for f in FEATURES:
        val   = input_vals[f["name"]]
        fmax  = f["max"]
        pct   = val / fmax if fmax else 0
        color, _ = risk_color(f["name"], val)
        st.markdown(f"""
        <div style="margin-bottom:0.6rem">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px">
            <span>{f['name']}</span>
            <span style="color:{color};font-weight:600">
              {val} {f['unit']} &nbsp;<span style="color:#7d8590;font-weight:400">(normal: {f['normal']})</span>
            </span>
          </div>
          <div style="background:#30363d;border-radius:4px;height:7px">
            <div style="width:{min(pct*100,100):.0f}%;background:{color};height:7px;border-radius:4px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

# ── Load ─────────────────────────────────────────────────────────
models, scaler, metrics = load_models()

# ── Hero ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>GlucoLens</h1>
  <p>Diabetes Risk Predictor · PIMA Indians Dataset · 6 ML Algorithms</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    algo = st.selectbox("Algorithm", list(models.keys()), index=0)
    st.markdown("---")
    st.markdown("### 📋 Normal Ranges")
    for f in FEATURES:
        if f["normal"] != "—":
            st.caption(f"**{f['name']}:** {f['normal']}")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not a medical diagnosis tool.")

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🩸 Predict", "📊 Model Comparison"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown('<div class="section-header">Enter Patient Data</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("📄 Upload lab report (PDF or TXT) — auto-fills inputs", type=["pdf", "txt"])
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

        predict_btn    = st.button("🔍 Predict Diabetes Risk", use_container_width=True)
        all_models_btn = st.button("⚡ Run All Models",        use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Result</div>', unsafe_allow_html=True)

        if predict_btn:
            arr        = np.array([[input_vals[n] for n in FEATURE_NAMES]])
            arr_scaled = scaler.transform(arr)
            m          = models[algo]
            pred       = int(m.predict(arr_scaled)[0])

            conf_str = ""
            if hasattr(m, "predict_proba"):
                try:
                    prob = m.predict_proba(arr_scaled)[0]
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
                  <p class="result-sub" style="font-size:0.75rem;margin-top:1rem">Algorithm: {algo}</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-risk">
                  <div style="font-size:3rem">⚠️</div>
                  <div class="result-title" style="color:#ef4444">Diabetic Risk Detected</div>
                  <p class="result-sub">Patient shows signs of diabetes</p>
                  {conf_str}
                  <p class="result-sub" style="font-size:0.75rem;margin-top:1rem">Algorithm: {algo}</p>
                </div>""", unsafe_allow_html=True)

            show_risk_gauge(input_vals)
            show_risk_bars(input_vals)

        elif all_models_btn:
            arr        = np.array([[input_vals[n] for n in FEATURE_NAMES]])
            arr_scaled = scaler.transform(arr)

            st.markdown('<div class="section-header">All Models — Side by Side</div>', unsafe_allow_html=True)

            all_preds = {}
            for name, m in models.items():
                p    = int(m.predict(arr_scaled)[0])
                conf = None
                if hasattr(m, "predict_proba"):
                    try:
                        prob = m.predict_proba(arr_scaled)[0]
                        conf = prob[p] * 100
                    except: pass
                all_preds[name] = {"pred": p, "conf": conf}

            votes_diabetic = sum(1 for v in all_preds.values() if v["pred"] == 1)
            votes_not      = len(all_preds) - votes_diabetic
            majority       = "Diabetic" if votes_diabetic > votes_not else "Not Diabetic"
            maj_color      = "#ef4444" if majority == "Diabetic" else "#22c55e"
            maj_icon       = "⚠️" if majority == "Diabetic" else "✅"
            agree          = votes_diabetic == len(all_preds) or votes_not == len(all_preds)
            agreement_text = "All models agree" if agree else f"{max(votes_diabetic, votes_not)}/{len(all_preds)} models agree"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#161b22,#1c2128);
                        border:2px solid {maj_color};border-radius:14px;
                        padding:1.2rem 1.5rem;text-align:center;margin-bottom:1.2rem">
              <div style="font-size:2rem">{maj_icon}</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.6rem;color:{maj_color}">{majority}</div>
              <div style="color:#7d8590;font-size:0.85rem;margin-top:0.3rem">{agreement_text} · {votes_not} Not Diabetic · {votes_diabetic} Diabetic</div>
            </div>""", unsafe_allow_html=True)

            for name, result in all_preds.items():
                p        = result["pred"]
                conf     = result["conf"]
                label    = "✅ Not Diabetic" if p == 0 else "⚠️ Diabetic"
                bcolor   = "#22c55e" if p == 0 else "#ef4444"
                bg       = "#052e16" if p == 0 else "#2d0a0a"
                conf_str = f" · {conf:.1f}% confidence" if conf else ""
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {bcolor}44;
                            border-left:4px solid {bcolor};border-radius:10px;
                            padding:0.9rem 1.2rem;margin-bottom:0.6rem;
                            display:flex;justify-content:space-between;align-items:center">
                  <div>
                    <span style="font-weight:600;color:#e6edf3">{name}</span>
                    <span style="color:#7d8590;font-size:0.8rem">{conf_str}</span>
                  </div>
                  <div style="color:{bcolor};font-weight:700">{label}</div>
                </div>""", unsafe_allow_html=True)

            if not agree:
                st.markdown("""
                <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:10px;
                            padding:0.9rem 1.2rem;margin-top:0.5rem;color:#f59e0b;font-size:0.85rem">
                  ⚠️ <b>Models disagree</b> — this case may be ambiguous.
                  Logistic Regression and Gradient Boosting are generally most reliable on this dataset.
                </div>""", unsafe_allow_html=True)

            show_risk_gauge(input_vals)
            show_risk_bars(input_vals)

        else:
            st.markdown("""
            <div style="background:#161b22;border:1px dashed #30363d;border-radius:16px;
                        padding:3rem;text-align:center;color:#7d8590;margin-top:2rem">
              <div style="font-size:3rem;margin-bottom:1rem">🩸</div>
              <p>Fill in the patient data and click <b>Predict</b> for one model<br>
              or <b>Run All Models</b> to compare all 6 at once</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Algorithm Performance Comparison</div>', unsafe_allow_html=True)

    df_m = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Algorithm"})

    cols = st.columns(4)
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    for i, lbl in enumerate(metric_labels):
        best_val  = df_m[lbl].max()
        best_algo = df_m.loc[df_m[lbl].idxmax(), "Algorithm"]
        cols[i].markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:#3b82f6">{best_val}%</div>
          <div class="lbl">Best {lbl}</div>
          <div style="color:#7d8590;font-size:0.75rem;margin-top:0.3rem">{best_algo}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    styled = df_m.set_index("Algorithm").style \
        .background_gradient(cmap="RdYlGn", vmin=60, vmax=85) \
        .format("{:.2f}%")
    st.dataframe(styled, use_container_width=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    insights = [
        ("🏆 Best Accuracy",   "Logistic Regression achieves the highest accuracy — a strong baseline for tabular medical data."),
        ("🌲 Best Recall",     "Decision Tree has the highest Recall — it catches the most actual diabetic cases, critical in medical diagnosis."),
        ("⚡ Best Overall",    "Gradient Boosting offers the best balance between Precision and Recall across all metrics."),
        ("⚠️ Recall Priority", "In diabetes detection, Recall matters more than Precision — missing a diabetic patient is more dangerous than a false alarm."),
        ("🏥 Recommendation",  "Use Gradient Boosting for deployment — best overall balance. Pair with Logistic Regression for explainability."),
    ]
    for icon_title, body in insights:
        st.markdown(f"""
        <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:0.6rem">
          <b style="color:#3b82f6">{icon_title}</b><br>
          <span style="color:#b0b5c8;font-size:0.9rem">{body}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:12px;
                padding:1.2rem;margin-top:1rem;color:#f59e0b;font-size:0.85rem">
      ⚠️ <b>Disclaimer:</b> This application is for educational purposes only and does not constitute
      medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.
    </div>""", unsafe_allow_html=True)
