"""
CardioRisk AI — Flask Application (SHAP-Explainable Edition)
===============================================================
Models used (per user selection):
  - CVD:      Logistic Regression (Baseline)  →  shap.LinearExplainer
  - Diabetes: Neural Network / MLP (Keras)    →  shap.KernelExplainer

Every prediction returns a live, per-user SHAP breakdown showing
exactly which input features pushed the risk up or down — not just
a static global importance chart.

NOTE ON KERAS VERSION (read this if you touch the diabetes model):
  We deliberately use plain `tensorflow.keras` (Keras 3, whatever ships
  with your installed TF) and shap.KernelExplainer — NOT tf_keras and
  NOT shap.GradientExplainer.

  GradientExplainer internally calls tf.keras.backend.learning_phase(),
  which no longer exists in Keras 3, so it crashes. The "fix" of forcing
  legacy Keras via tf_keras just trades that crash for a worse one,
  because .keras files saved by Keras 3 use an InputLayer config
  (batch_shape) that Keras 2 / tf_keras cannot deserialize at all.

  KernelExplainer sidesteps all of this: it treats the model as a black
  box and only ever calls .predict() on it, so it works identically
  regardless of which Keras version saved/loads the model. It's slower
  per call than GradientExplainer, which is why we shrink the SHAP
  background sample (see DIAB_SHAP_BG_SMALL below) to keep latency
  reasonable for a live, per-request explanation.
"""

import json, os, pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "cardiorisk-dev-key-change-in-prod")
BASE = os.path.dirname(os.path.abspath(__file__))

# Lazy/guarded import — TensorFlow is heavy and only needed for the
# diabetes model, so we import it here and fail with a clear message
# if it's missing rather than crashing on the CVD-only routes too.
try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for the diabetes Neural Network model. "
        "Run: pip install tensorflow"
    ) from e

import shap

# ── Load Diabetes artifacts (saved_artifacts_diabetes/) ───────────────────────
# NOTE: the diabetes model is now a Keras MLP (neural_network_mlp.keras),
# NOT a pickled sklearn/XGBoost model. Keras models are loaded with
# tf.keras.models.load_model(), never pickle.load().
D_DIR = os.path.join(BASE, "model", "diabetes")
DIAB_MODEL = tf.keras.models.load_model(f"{D_DIR}/neural_network_mlp.keras")

with open(f"{D_DIR}/scaler.pkl",            "rb") as f: DIAB_SCALER     = pickle.load(f)
with open(f"{D_DIR}/le_gender.pkl",         "rb") as f: LE_GENDER       = pickle.load(f)
with open(f"{D_DIR}/le_family.pkl",         "rb") as f: LE_FAMILY       = pickle.load(f)
with open(f"{D_DIR}/activity_map.pkl",      "rb") as f: ACTIVITY_MAP    = pickle.load(f)
with open(f"{D_DIR}/feature_columns.pkl",   "rb") as f: DIAB_FEATURES   = pickle.load(f)

# SHAP needs a background dataset baked into the explainer at creation
# time. That background sample was saved separately in the notebook
# patch as shap_background.pkl — a small numpy array of already-scaled
# training rows.
with open(f"{D_DIR}/shap_background.pkl",  "rb") as f: DIAB_SHAP_BG    = pickle.load(f)

try:
    with open(f"{D_DIR}/shap_importance.json") as f: DIAB_SHAP_GLOBAL = json.load(f)
except FileNotFoundError:
    DIAB_SHAP_GLOBAL = {}
try:
    with open(f"{D_DIR}/diab_metrics.json") as f: DIAB_METRICS = json.load(f)
except FileNotFoundError:
    DIAB_METRICS = {}


def predict_diabetes_score(X_scaled):
    """
    Wraps the Keras model's predict() call.
    Keras returns shape (n_samples, 1) — sklearn/XGBoost return (n_samples,).
    .flatten() normalises this so downstream code doesn't need to know
    which model type is behind it. verbose=0 silences the per-call
    progress bar that Keras prints by default.
    """
    raw_pred = DIAB_MODEL.predict(X_scaled, verbose=0).flatten()[0]
    return float(np.clip(raw_pred, 0, 100))


def _diab_predict_fn(X):
    """
    Black-box predict function for shap.KernelExplainer.
    KernelExplainer perturbs rows and calls this repeatedly, so it must
    accept a 2D numpy array and return a 1D array of model outputs.
    verbose=0 avoids Keras spamming the console on every perturbation.
    """
    X = np.asarray(X, dtype=np.float32)
    return DIAB_MODEL.predict(X, verbose=0).flatten()


# Build the SHAP explainer once at startup (not per-request — this is
# the expensive part). KernelExplainer only calls .predict() under the
# hood, so it works regardless of Keras version / model internals.
#
# KernelExplainer cost scales with background-sample size, so we shrink
# it from whatever was saved (could be hundreds/thousands of rows) down
# to a small representative sample. 50 rows is a common, reasonable
# default for live, per-request explanations.
if len(DIAB_SHAP_BG) > 50:
    DIAB_SHAP_BG_SMALL = shap.sample(DIAB_SHAP_BG, 50)
else:
    DIAB_SHAP_BG_SMALL = DIAB_SHAP_BG

DIAB_EXPLAINER = shap.KernelExplainer(_diab_predict_fn, DIAB_SHAP_BG_SMALL)

# ── Load CVD artifacts (saved_artifacts/) ─────────────────────────────────────
C_DIR = os.path.join(BASE, "model", "cvd")
with open(f"{C_DIR}/logistic_regression_baseline.pkl", "rb") as f: CVD_MODEL      = pickle.load(f)
with open(f"{C_DIR}/scaler.pkl",                        "rb") as f: CVD_SCALER     = pickle.load(f)
with open(f"{C_DIR}/scale_cols.pkl",                    "rb") as f: CVD_SCALE_COLS = pickle.load(f)
with open(f"{C_DIR}/ordinal_encoder.pkl",               "rb") as f: CVD_ORD_ENC    = pickle.load(f)
with open(f"{C_DIR}/ohe_columns.pkl",                   "rb") as f: CVD_OHE_COLS   = pickle.load(f)
with open(f"{C_DIR}/top_features.pkl",                  "rb") as f: CVD_FEATURES   = pickle.load(f)
with open(f"{C_DIR}/shap_explainer.pkl",                "rb") as f: CVD_EXPLAINER  = pickle.load(f)
try:
    with open(f"{C_DIR}/diabetes_raw_values.pkl", "rb") as f: CVD_DIAB_RAW_VALUES = pickle.load(f)
except FileNotFoundError:
    CVD_DIAB_RAW_VALUES = [c.replace("Diabetes_", "") for c in CVD_OHE_COLS]
try:
    with open(f"{C_DIR}/shap_importance.json") as f: CVD_SHAP_GLOBAL = json.load(f)
except FileNotFoundError:
    CVD_SHAP_GLOBAL = {}
try:
    with open(f"{C_DIR}/cvd_metrics.json") as f: CVD_METRICS = json.load(f)
except FileNotFoundError:
    CVD_METRICS = {}

# ── Constants matching notebook encoding orders exactly ───────────────────────
GENERAL_HEALTH_ORDER = ["Poor", "Fair", "Good", "Very Good", "Excellent"]
AGE_ORDER = ["18-24","25-29","30-34","35-39","40-44","45-49",
             "50-54","55-59","60-64","65-69","70-74","75-79","80+"]
CHECKUP_ORDER = [
    "Never",
    "Within the past 5 years (2 years but less than 5 years ago)",
    "Within the past 2 years (1 year but less than 2 years ago)",
    "Within the past year (anytime less than 12 months ago)",
]
BINARY_LABEL_MAP = {"No": 0, "Yes": 1}

# Human-readable labels for SHAP driver display
FEATURE_LABELS = {
    "General_Health": "General Health Rating",
    "Age_Category": "Age Group",
    "Checkup": "Last Checkup",
    "BMI": "Body Mass Index",
    "Smoking_History": "Smoking History",
    "Exercise": "Physical Exercise",
    "Alcohol_Consumption": "Alcohol Consumption",
    "Fruit_Consumption": "Fruit Consumption",
    "Green_Vegetables_Consumption": "Green Vegetable Intake",
    "FriedPotato_Consumption": "Fried Potato Intake",
    "Comorbidity_Score": "Comorbidity Score",
    "Sex": "Sex",
    "Skin_Cancer": "Skin Cancer History",
    "Other_Cancer": "Other Cancer History",
    "Depression": "Depression History",
    "Arthritis": "Arthritis",
    "Height_(cm)": "Height",
    "Weight_(kg)": "Weight",
    "age": "Age",
    "gender": "Gender",
    "bmi": "Body Mass Index",
    "blood_pressure": "Blood Pressure",
    "fasting_glucose_level": "Fasting Glucose",
    "insulin_level": "Insulin Level",
    "HbA1c_level": "HbA1c Level",
    "cholesterol_level": "Cholesterol",
    "triglycerides_level": "Triglycerides",
    "physical_activity_level": "Physical Activity",
    "daily_calorie_intake": "Daily Calorie Intake",
    "sugar_intake_grams_per_day": "Sugar Intake",
    "sleep_hours": "Sleep Hours",
    "stress_level": "Stress Level",
    "family_history_diabetes": "Family History of Diabetes",
    "waist_circumference_cm": "Waist Circumference",
}
def label_for(feat):
    return FEATURE_LABELS.get(feat, feat.replace("_", " "))


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_bmi(height_cm, weight_kg):
    try:
        h, w = float(height_cm), float(weight_kg)
        return round(w / ((h / 100) ** 2), 2) if h > 0 and w > 0 else None
    except (TypeError, ValueError):
        return None

def bmi_category(bmi):
    if bmi is None:  return "Unknown"
    if bmi < 18.5:   return "Underweight"
    elif bmi < 25:   return "Normal weight"
    elif bmi < 30:   return "Overweight"
    elif bmi < 35:   return "Obese (Class I)"
    elif bmi < 40:   return "Obese (Class II)"
    else:            return "Obese (Class III)"

def bmi_color(bmi):
    if bmi is None: return "#94a3b8"
    if bmi < 18.5:  return "#60a5fa"
    elif bmi < 25:  return "#22c55e"
    elif bmi < 30:  return "#f59e0b"
    else:           return "#ef4444"

def diabetes_score_to_category(score):
    if score < 35:   return {"cat": "Low Risk",    "color": "#22c55e"}
    elif score < 65: return {"cat": "Prediabetes", "color": "#f59e0b"}
    else:            return {"cat": "High Risk",   "color": "#ef4444"}

def diabetes_score_to_cvd_value(score, gestational=False):
    """Maps the diabetes regression score -> one of the 4 raw CVD Diabetes strings."""
    if gestational:
        # Find the raw value containing 'pregnancy'
        for v in CVD_DIAB_RAW_VALUES:
            if "pregnancy" in v.lower():
                return v
        return "Yes, but female told only during pregnancy"
    if score >= 65:
        for v in CVD_DIAB_RAW_VALUES:
            if v.strip() == "Yes":
                return v
        return "Yes"
    if score >= 35:
        for v in CVD_DIAB_RAW_VALUES:
            if "pre-diabetes" in v.lower() or "borderline" in v.lower():
                return v
        return "No, pre-diabetes or borderline diabetes"
    return "No"

def cvd_risk_label(prob):
    if prob < 0.15:   return {"label": "Low Risk",       "color": "#22c55e"}
    elif prob < 0.35: return {"label": "Moderate Risk",  "color": "#f59e0b"}
    elif prob < 0.60: return {"label": "High Risk",      "color": "#ef4444"}
    else:             return {"label": "Very High Risk", "color": "#7f1d1d"}

def safe_float(v, default=np.nan):
    try:    return float(v) if v not in (None, "", "None") else default
    except (TypeError, ValueError): return default


# ── Diabetes preprocessing (mirrors the notebook exactly) ─────────────────────
def preprocess_diabetes(raw, bmi):
    row = {col: safe_float(raw.get(col)) for col in [
        "age", "blood_pressure", "fasting_glucose_level", "insulin_level",
        "HbA1c_level", "cholesterol_level", "triglycerides_level",
        "daily_calorie_intake", "sugar_intake_grams_per_day",
        "sleep_hours", "stress_level", "waist_circumference_cm"]}
    row["bmi"] = bmi

    gender_raw = str(raw.get("gender", "")).strip().lower()
    try:    row["gender"] = int(LE_GENDER.transform([gender_raw])[0])
    except: row["gender"] = 0

    row["physical_activity_level"] = ACTIVITY_MAP.get(
        raw.get("physical_activity_level", "Low"), 0)

    fam_raw = str(raw.get("family_history_diabetes", "No")).strip()
    try:    row["family_history_diabetes"] = int(LE_FAMILY.transform([fam_raw])[0])
    except: row["family_history_diabetes"] = 0

    df_row = pd.DataFrame([{col: row.get(col, np.nan) for col in DIAB_FEATURES}])
    scaled = DIAB_SCALER.transform(df_row)
    return scaled, df_row


# ── CVD preprocessing (mirrors the notebook exactly) ───────────────────────────
def preprocess_cvd(raw, bmi, diabetes_value):
    d = {}
    d["Height_(cm)"] = safe_float(raw.get("height_cm"))
    d["Weight_(kg)"] = safe_float(raw.get("weight_kg"))
    d["BMI"]         = bmi
    d["Alcohol_Consumption"]          = np.log1p(safe_float(raw.get("Alcohol_Consumption"), 0))
    d["Fruit_Consumption"]            = np.log1p(safe_float(raw.get("Fruit_Consumption"), 0))
    d["Green_Vegetables_Consumption"] = np.log1p(safe_float(raw.get("Green_Vegetables_Consumption"), 0))
    d["FriedPotato_Consumption"]      = np.log1p(safe_float(raw.get("FriedPotato_Consumption"), 0))

    yn = lambda k: BINARY_LABEL_MAP.get(str(raw.get(k, "No")).strip(), 0)
    d["Exercise"]        = yn("Exercise")
    d["Skin_Cancer"]     = yn("Skin_Cancer")
    d["Other_Cancer"]    = yn("Other_Cancer")
    d["Depression"]      = yn("Depression")
    d["Arthritis"]       = yn("Arthritis")
    d["Smoking_History"] = yn("Smoking_History")
    sex_raw = str(raw.get("Sex", "Female")).strip()
    d["Sex"] = 1 if sex_raw.lower() == "male" else 0

    gh  = str(raw.get("General_Health", "Good")).strip()
    age = str(raw.get("Age_Category", "45-49")).strip()
    chk = str(raw.get("Checkup", CHECKUP_ORDER[-1])).strip()
    d["General_Health"] = (GENERAL_HEALTH_ORDER.index(gh)  if gh  in GENERAL_HEALTH_ORDER  else 2) + 1
    d["Age_Category"]   = (AGE_ORDER.index(age)             if age in AGE_ORDER              else 5) + 1
    d["Checkup"]        = (CHECKUP_ORDER.index(chk)        if chk in CHECKUP_ORDER          else 3) + 1

    for col in CVD_OHE_COLS:
        d[col] = 0
    ohe_key = f"Diabetes_{diabetes_value}"
    if ohe_key in d:
        d[ohe_key] = 1
    else:
        # fall back to 'No' if the exact string doesn't match any OHE column
        fallback_key = "Diabetes_No"
        if fallback_key in d:
            d[fallback_key] = 1

    diab_binary = 0 if diabetes_value == "No" else 1
    d["Comorbidity_Score"] = (
        diab_binary + d["Arthritis"] + d["Skin_Cancer"] +
        d["Other_Cancer"] + d["Depression"]
    )

    all_cols = (["Height_(cm)", "Weight_(kg)", "BMI",
                 "Alcohol_Consumption", "Fruit_Consumption",
                 "Green_Vegetables_Consumption", "FriedPotato_Consumption",
                 "Exercise", "Skin_Cancer", "Other_Cancer", "Depression",
                 "Arthritis", "Sex", "Smoking_History", "General_Health",
                 "Age_Category", "Checkup", "Comorbidity_Score"]
                + CVD_OHE_COLS)

    df_row = pd.DataFrame([{col: d.get(col, 0) for col in all_cols}])

    scale_present = [c for c in CVD_SCALE_COLS if c in df_row.columns]
    df_row[scale_present] = CVD_SCALER.transform(df_row[scale_present])

    df_sel = df_row[CVD_FEATURES]
    return df_sel.values, df_sel


def build_shap_drivers(explainer, X_scaled, feature_names, top_n=8, label_map=None):
    """
    Computes a per-user SHAP explanation and returns a sorted list of
    {feature, shap_value, direction} for the top_n most influential features.

    Works across the explainer types used in this app:
      - shap.LinearExplainer   (CVD — Logistic Regression)
      - shap.KernelExplainer   (Diabetes — Keras MLP, model-agnostic)

    KernelExplainer (and LinearExplainer for binary classifiers) can
    return either a plain (n_samples, n_features) array or a list of
    per-class/per-output arrays, so we normalise both shapes here:
      - list            -> take the last entry (e.g. "positive class" or
                            the single regression output)
      - (1, n_features)  -> single row, single output (our case)
    """
    X_input = np.asarray(X_scaled, dtype=np.float32)
    shap_vals = explainer.shap_values(X_input)

    if isinstance(shap_vals, list):       # e.g. per-class outputs
        shap_vals = shap_vals[-1]

    shap_vals = np.array(shap_vals).reshape(-1)

    pairs = list(zip(feature_names, shap_vals))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = pairs[:top_n]

    max_abs = max([abs(v) for _, v in top], default=0.0001) or 0.0001

    drivers = []
    for feat, val in top:
        drivers.append({
            "feature":   feat,
            "label":     (label_map.get(feat, feat.replace("_", " "))
                          if label_map else feat.replace("_", " ")),
            "shap_value": round(float(val), 4),
            "direction": "increases" if val > 0 else "decreases",
            "magnitude_pct": round(abs(val) / max_abs * 100, 1),
        })
    return drivers


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           cvd_metrics=CVD_METRICS, diab_metrics=DIAB_METRICS)

@app.route("/diabetes")
def diabetes_page():
    return render_template("diabetes.html", diab_shap=DIAB_SHAP_GLOBAL)

@app.route("/cvd")
def cvd_page():
    prefill = session.get("diabetes_result", {})
    return render_template("cvd.html", prefill=prefill, cvd_shap=CVD_SHAP_GLOBAL)

@app.route("/simulator")
def simulator_page():
    prefill = session.get("diabetes_result", {})
    return render_template("simulator.html", prefill=prefill)

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html",
                           cvd_metrics=CVD_METRICS, diab_metrics=DIAB_METRICS,
                           cvd_shap=CVD_SHAP_GLOBAL, diab_shap=DIAB_SHAP_GLOBAL)


# ── API: Diabetes predict (with live SHAP) ────────────────────────────────────
@app.route("/api/predict/diabetes", methods=["POST"])
def api_predict_diabetes():
    try:
        raw = request.get_json(force=True)
        bmi = compute_bmi(raw.get("height_cm"), raw.get("weight_kg"))
        if not bmi:
            return jsonify({"success": False, "error": "Invalid height/weight"}), 400

        X_scaled, X_df = preprocess_diabetes(raw, bmi)
        score = predict_diabetes_score(X_scaled)
        cat   = diabetes_score_to_category(score)

        # Live per-user SHAP explanation
        drivers = build_shap_drivers(
            DIAB_EXPLAINER, X_scaled, DIAB_FEATURES,
            top_n=8, label_map=FEATURE_LABELS
        )

        is_female = str(raw.get("gender", "")).strip().lower() == "female"

        session["diabetes_result"] = {
            "score": score, "category": cat["cat"], "color": cat["color"],
            "bmi": bmi, "bmi_category": bmi_category(bmi),
            "height_cm": safe_float(raw.get("height_cm")),
            "weight_kg": safe_float(raw.get("weight_kg")),
            "is_female": is_female, "gestational": False,
            "diabetes_value": diabetes_score_to_cvd_value(score, False),
        }
        session.modified = True

        return jsonify({
            "success": True, "score": round(score, 2), "category": cat,
            "bmi": bmi, "bmi_category": bmi_category(bmi), "bmi_color": bmi_color(bmi),
            "is_female": is_female, "show_gest_q": is_female and score >= 35,
            "shap_drivers": drivers,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/update-gestational", methods=["POST"])
def api_update_gestational():
    try:
        raw  = request.get_json(force=True)
        gest = bool(raw.get("gestational_only", False))
        if "diabetes_result" not in session:
            return jsonify({"success": False, "error": "No session"}), 400
        score = session["diabetes_result"]["score"]
        session["diabetes_result"]["gestational"]    = gest
        session["diabetes_result"]["diabetes_value"] = diabetes_score_to_cvd_value(score, gest)
        session.modified = True
        return jsonify({"success": True,
                        "diabetes_value": session["diabetes_result"]["diabetes_value"]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ── API: CVD predict (with live SHAP) ──────────────────────────────────────────
@app.route("/api/predict/cvd", methods=["POST"])
def api_predict_cvd():
    try:
        raw  = request.get_json(force=True)
        sess = session.get("diabetes_result", {})

        height = raw.get("height_cm") or sess.get("height_cm")
        weight = raw.get("weight_kg") or sess.get("weight_kg")
        bmi    = compute_bmi(height, weight)
        if not bmi:
            return jsonify({"success": False, "error": "Invalid height/weight"}), 400

        diabetes_value = raw.get("diabetes_value") or sess.get("diabetes_value") or "No"

        X_arr, X_df = preprocess_cvd(raw, bmi, diabetes_value)
        prob = float(CVD_MODEL.predict_proba(X_arr)[0][1])
        risk = cvd_risk_label(prob)

        # Live per-user SHAP explanation
        drivers = build_shap_drivers(
            CVD_EXPLAINER, X_arr, CVD_FEATURES,
            top_n=8, label_map=FEATURE_LABELS
        )

        return jsonify({
            "success": True, "probability": round(prob, 4),
            "prob_pct": round(prob * 100, 1), "predicted": int(prob >= 0.5),
            "risk": risk, "bmi": bmi, "bmi_category": bmi_category(bmi),
            "bmi_color": bmi_color(bmi), "diabetes_used": diabetes_value,
            "from_session": bool(sess), "shap_drivers": drivers,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ── API: Intervention simulator ────────────────────────────────────────────────
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        raw           = request.get_json(force=True)
        model_type    = raw.get("model", "cvd")
        base_inputs   = dict(raw.get("inputs", {}))
        interventions = raw.get("interventions", {})
        sess          = session.get("diabetes_result", {})

        def score_diab(inp):
            h = inp.get("height_cm") or sess.get("height_cm")
            w = inp.get("weight_kg") or sess.get("weight_kg")
            bmi = compute_bmi(h, w) or 25.0
            X_scaled, _ = preprocess_diabetes(inp, bmi)
            return predict_diabetes_score(X_scaled)

        def score_cvd(inp):
            h = inp.get("height_cm") or sess.get("height_cm")
            w = inp.get("weight_kg") or sess.get("weight_kg")
            bmi = compute_bmi(h, w) or 25.0
            dv = inp.get("diabetes_value") or sess.get("diabetes_value") or "No"
            X_arr, _ = preprocess_cvd(inp, bmi, dv)
            return float(CVD_MODEL.predict_proba(X_arr)[0][1])

        get_score = score_cvd if model_type == "cvd" else score_diab
        baseline  = get_score(base_inputs)
        new_score = get_score({**base_inputs, **interventions})
        delta     = new_score - baseline
        pct       = delta / max(abs(baseline), 0.001) * 100

        marginals = {}
        for feat, val in interventions.items():
            s = get_score({**base_inputs, feat: val})
            marginals[feat] = {
                "new_score": round(s, 4),
                "delta":     round(s - baseline, 4),
                "pct":       round((s - baseline) / max(abs(baseline), 0.001) * 100, 2),
            }

        return jsonify({
            "success": True, "baseline": round(baseline, 4),
            "new_score": round(new_score, 4), "delta": round(delta, 4),
            "pct_change": round(pct, 2), "marginals": marginals,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)