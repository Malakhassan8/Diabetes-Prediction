# 🩸 GlucoLens — Diabetes Risk Predictor

A Streamlit web app that predicts diabetes risk using a Decision Tree classifier trained on the PIMA Indians Diabetes Dataset.

---

## 🚀 Live Demo
👉 **[https://diabetes-prediction-jewhvuhnwlyng64sg3mckr.streamlit.app/](https://diabetes-prediction-jewhvuhnwlyng64sg3mckr.streamlit.app/)** 

---

## ✨ Features

- 📄 **PDF / TXT Upload** — upload a lab report and inputs auto-fill automatically
- 🔢 **Manual Input** — number inputs for all 8 clinical features with unit labels
- 📊 **Risk Gauge** — semicircle meter showing Low / Moderate / High risk
- 📉 **Feature Risk Bars** — color-coded bars (green / yellow / red) showing which indicators are above normal
- 🏥 **Normal Ranges** — healthy range shown next to every input for patient context
- ⚡ **No file dependencies** — model trains automatically at startup, no `.sav` files needed

---

## 🧠 Model

| Property | Detail |
|---|---|
| Algorithm | Decision Tree (Entropy criterion, max depth 5) |
| Dataset | PIMA Indians Diabetes Dataset (768 samples) |
| Features | 8 clinical indicators |
| Target | Diabetic (1) / Not Diabetic (0) |
| Train/Test Split | 80% / 20% |
| Preprocessing | StandardScaler |

---

## 📋 Features Used

| Feature | Description | Normal Range |
|---|---|---|
| Pregnancies | Number of times pregnant | 0–5 |
| Glucose | Plasma glucose concentration | 70–100 mg/dL |
| Blood Pressure | Diastolic blood pressure | 60–80 mm Hg |
| Skin Thickness | Triceps skinfold thickness | 10–30 mm |
| Insulin | 2-Hour serum insulin | 16–166 µU/mL |
| BMI | Body mass index | 18.5–24.9 kg/m² |
| Diabetes Pedigree Function | Genetic risk score | 0.0–0.5 |
| Age | Age in years | — |

---

## 🖥️ Run Locally

```bash
git clone https://github.com/yourusername/diabetes-predictor
cd diabetes-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set main file to `app.py`
5. Click **Deploy** — live in ~2 minutes

---

## 📁 Project Structure

```
diabetes-predictor/
├── app.py            ← Main Streamlit app
├── requirements.txt  ← Python dependencies
└── README.md         ← You are here
```

---

## 🔗 Related Project

Also check out my **Breast Cancer Classifier** — classifies tumors as Benign or Malignant using 4 ML algorithms with PDF upload and side-by-side model comparison.

👉 [github.com/yourusername/breast-cancer-classifier](#) ← replace with your link

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and does not constitute medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.

---

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/yourprofile](#)
- GitHub: [github.com/yourusername](#)
