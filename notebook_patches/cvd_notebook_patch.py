# ════════════════════════════════════════════════════════════════
# ADD THIS CELL AT THE END OF YOUR CVD NOTEBOOK
# (after the "Saving Preprocessing Objects and Models" section)
# ════════════════════════════════════════════════════════════════
#
# Why this is needed:
#   1. You saved `logistic_regression_baseline.pkl` but never saved
#      the Diabetes one-hot column NAMES with their exact string
#      values — only `ohe_columns.pkl` which has the column names,
#      but Flask also needs to know which 4 raw string values map
#      to them. This cell saves that explicitly.
#   2. SHAP for Logistic Regression uses shap.LinearExplainer
#      (NOT TreeExplainer, which only works for tree models).
#      This was never computed in your original notebook.
#   3. The web app needs feature importances in a portable JSON
#      format, not just a fitted explainer object.

import shap
import json
import numpy as np

print("Computing SHAP values for Logistic Regression (baseline)...")

# LogisticRegression is a linear model -> use LinearExplainer
# X_train_sel is the background data SHAP uses to estimate the
# expected value (baseline). Using a sample for speed.
background = X_train_sel.sample(min(500, len(X_train_sel)), random_state=42)

explainer_lr = shap.LinearExplainer(logreg_model, background)
shap_values_lr = explainer_lr.shap_values(X_test_sel)

# Mean absolute SHAP value per feature = global importance
shap_importance_cvd = dict(zip(
    top_features,
    np.abs(shap_values_lr).mean(axis=0).tolist()
))

# Sort descending for readability
shap_importance_cvd = dict(
    sorted(shap_importance_cvd.items(), key=lambda x: -x[1])
)

print("\nTop 10 SHAP features (Logistic Regression):")
for feat, val in list(shap_importance_cvd.items())[:10]:
    print(f"  {feat:35s}: {val:.4f}")

# Visualise
shap.summary_plot(shap_values_lr, X_test_sel, max_display=15)

# ── Save SHAP importances as JSON (portable, no pickle needed) ───
with open(f'{SAVE_DIR}/shap_importance.json', 'w') as f:
    json.dump(shap_importance_cvd, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/shap_importance.json")

# ── Save the SHAP explainer itself too (for live per-user SHAP) ──
# This lets Flask compute a SHAP waterfall for each individual
# prediction, not just the global average.
with open(f'{SAVE_DIR}/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer_lr, f)
print(f"Saved: {SAVE_DIR}/shap_explainer.pkl")

# ── Save Diabetes OHE value -> column name mapping explicitly ────
# Your existing ohe_columns.pkl has the column NAMES
# (e.g. 'Diabetes_Yes'), but the web app needs to know exactly
# what raw string the user-facing dropdown produces for each.
diabetes_raw_values = [c.replace('Diabetes_', '') for c in ohe_columns]
with open(f'{SAVE_DIR}/diabetes_raw_values.pkl', 'wb') as f:
    pickle.dump(diabetes_raw_values, f)
print(f"Saved: {SAVE_DIR}/diabetes_raw_values.pkl")
print(f"Diabetes raw values: {diabetes_raw_values}")

# ── Save model performance metrics for the dashboard ──────────────
cvd_metrics = {
    'model_name': 'Logistic Regression (Baseline)',
    'accuracy':   round(lr_results['Accuracy'], 4),
    'precision':  round(lr_results['Precision'], 4),
    'recall':     round(lr_results['Recall'], 4),
    'f1':         round(lr_results['F1'], 4),
    'roc_auc':    round(lr_results['ROC-AUC'], 4),
}
with open(f'{SAVE_DIR}/cvd_metrics.json', 'w') as f:
    json.dump(cvd_metrics, f, indent=2)
print(f"Saved: {SAVE_DIR}/cvd_metrics.json")
print(cvd_metrics)
