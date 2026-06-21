# ════════════════════════════════════════════════════════════════
# ADD THIS CELL AT THE END OF YOUR DIABETES NOTEBOOK
# (after the "Saving Preprocessing Objects and Models" section)
# ════════════════════════════════════════════════════════════════
#
# Why this is needed:
#   1. SHAP for XGBoost uses shap.TreeExplainer (exact and fast).
#      This was never computed in your original notebook.
#   2. The web app needs feature importances in portable JSON
#      format so it can render driver bars without re-running
#      the SHAP computation on every request.
#   3. We also save the live explainer object so Flask can compute
#      a per-user SHAP waterfall (not just the global average).

import shap
import json
import numpy as np

print("Computing SHAP values for XGBoost (Tuned)...")

# XGBoost is a tree model -> use TreeExplainer (exact, fast)
explainer_xgb = shap.TreeExplainer(xgb_model_tuned)
shap_values_xgb = explainer_xgb.shap_values(X_test_scaled)

# Mean absolute SHAP value per feature = global importance
shap_importance_diab = dict(zip(
    list(X.columns),
    np.abs(shap_values_xgb).mean(axis=0).tolist()
))

# Sort descending for readability
shap_importance_diab = dict(
    sorted(shap_importance_diab.items(), key=lambda x: -x[1])
)

print("\nTop 10 SHAP features (XGBoost Tuned):")
for feat, val in list(shap_importance_diab.items())[:10]:
    print(f"  {feat:35s}: {val:.4f}")

# Visualise
shap.summary_plot(shap_values_xgb,
                  pd.DataFrame(X_test_scaled, columns=X.columns),
                  max_display=15)

# ── Save SHAP importances as JSON ─────────────────────────────────
with open(f'{SAVE_DIR}/shap_importance.json', 'w') as f:
    json.dump(shap_importance_diab, f, indent=2)
print(f"\nSaved: {SAVE_DIR}/shap_importance.json")

# ── Save the SHAP explainer for live per-user SHAP in Flask ───────
with open(f'{SAVE_DIR}/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer_xgb, f)
print(f"Saved: {SAVE_DIR}/shap_explainer.pkl")

# ── Save model performance metrics for the dashboard ──────────────
diab_metrics = {
    'model_name': 'XGBoost Regressor (Tuned)',
    'mae':        round(xgb_tuned_results['MAE'], 4),
    'rmse':       round(xgb_tuned_results['RMSE'], 4),
    'r2':         round(xgb_tuned_results['R2'], 4),
    'mape':       round(xgb_tuned_results['MAPE (%)'], 2),
}
with open(f'{SAVE_DIR}/diab_metrics.json', 'w') as f:
    json.dump(diab_metrics, f, indent=2)
print(f"Saved: {SAVE_DIR}/diab_metrics.json")
print(diab_metrics)
