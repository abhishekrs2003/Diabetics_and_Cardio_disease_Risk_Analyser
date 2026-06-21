/* ============================================================
   CardioRisk AI — Prediction page JS (predict.js)
   ============================================================ */

let _isFemale      = false;
let _diabetesScore = 0;
let _gestational   = false;

// ── Gender change → may reveal gestational question ────────────
function onGenderChange() {
  const sel = document.querySelector('#diab_gender');
  _isFemale = sel && sel.value === 'female';
  maybeShowGestational();
}

function maybeShowGestational() {
  const el = document.getElementById('gestational_question');
  if (!el) return;
  const show = _isFemale && _diabetesScore >= 35;
  el.style.display = show ? 'block' : 'none';
  if (!show) {
    _gestational = false;
    document.querySelectorAll('input[name="gestational_diabetes"]').forEach(r => { r.checked = false; });
  }
}

async function onGestationalChange(val) {
  _gestational = (val === 'yes');
  await fetch('/api/update-gestational', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gestational_only: _gestational }),
  });
  updateNextStepLabel();
}

function updateNextStepLabel() {
  const el = document.getElementById('next_step_sub');
  if (!el) return;
  let label;
  if (_gestational)            label = 'Diabetes = Gestational only';
  else if (_diabetesScore >= 65) label = 'Diabetes = Yes (confirmed risk)';
  else if (_diabetesScore >= 35) label = 'Diabetes = Pre-diabetes / borderline';
  else                            label = 'Diabetes = No';
  el.textContent = label + ' will be prefilled in CVD';
}

/* ── SHAP plain-English summary generator ────────────────────
   Turns the top driver list into a readable sentence, e.g.
   "Your HbA1c Level and Fasting Glucose are the biggest factors
    pushing your risk up, while your Sleep Hours is helping lower it."
*/
function buildSHAPSummary(drivers, scoreLabel) {
  if (!drivers || !drivers.length) return '';

  const increasing = drivers.filter(d => d.direction === 'increases').slice(0, 2);
  const decreasing = drivers.filter(d => d.direction === 'decreases').slice(0, 2);

  let sentence = '';
  if (increasing.length) {
    const names = increasing.map(d => `<strong>${d.label}</strong>`).join(' and ');
    sentence += `Your ${names} ${increasing.length > 1 ? 'are' : 'is'} the biggest factor${increasing.length > 1 ? 's' : ''} pushing your ${scoreLabel} up. `;
  }
  if (decreasing.length) {
    const names = decreasing.map(d => `<strong>${d.label}</strong>`).join(' and ');
    sentence += `Your ${names} ${decreasing.length > 1 ? 'are' : 'is'} helping lower it.`;
  }
  return sentence || 'No single factor stands out strongly — your inputs are fairly balanced.';
}

// ── Body change handler (BMI card update) ──────────────────────
// onBodyChange(prefix) is defined in utils.js

// ── Diabetes form submit ────────────────────────────────────────
async function submitDiabetes() {
  const btn = document.getElementById('btn_predict_diab');
  if (!btn) return;
  btn.classList.add('loading');
  btn.querySelector('.btn-spinner')?.classList.add('spinning');
  btn.disabled = true;

  try {
    const data = collectForm('diabetes_form');
    const bmi = calcBMI(data.height_cm, data.weight_kg);
    if (!bmi) { showError('diab_error', 'Please enter valid height and weight.'); return; }
    if (bmi < 10 || bmi > 80) { showError('diab_error', 'Height or weight values seem incorrect.'); return; }

    clearError('diab_error');
    const res    = await fetch('/api/predict/diabetes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const result = await res.json();
    if (!result.success) throw new Error(result.error);

    _diabetesScore = result.score;
    _isFemale      = result.is_female;

    showDiabetesResult(result);
    maybeShowGestational();

  } catch (e) {
    showError('diab_error', e.message);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-spinner')?.classList.remove('spinning');
    btn.disabled = false;
  }
}

function showDiabetesResult(r) {
  document.getElementById('diab_placeholder')?.classList.add('hidden');
  document.getElementById('diab_placeholder').style.display = 'none';
  const content = document.getElementById('diab_result');
  if (content) content.style.display = 'block';

  animateGauge('diab_gauge_arc', 'diab_gauge_text', r.score, 100, r.category.color);

  const badge = document.getElementById('diab_risk_badge');
  if (badge) {
    badge.textContent = r.category.cat;
    badge.style.color = r.category.color;
  }

  const bmiVal = document.getElementById('diab_bmi_val');
  const bmiCat = document.getElementById('diab_bmi_cat');
  if (bmiVal) { bmiVal.textContent = r.bmi?.toFixed(1) ?? '—'; bmiVal.style.color = r.bmi_color; }
  if (bmiCat) { bmiCat.textContent = r.bmi_category ?? '';     bmiCat.style.color = r.bmi_color; }

  // Live SHAP waterfall from the server (per-user, not global)
  renderSHAPWaterfall('diab_shap_waterfall', r.shap_drivers);
  const summaryEl = document.getElementById('diab_shap_summary');
  if (summaryEl) summaryEl.innerHTML = buildSHAPSummary(r.shap_drivers, 'diabetes risk score');

  updateNextStepLabel();
  const nxt = document.getElementById('diab_next_btn');
  if (nxt) nxt.style.display = 'flex';

  if (window.innerWidth < 900) content.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── CVD form submit ───────────────────────────────────────────
async function submitCVD() {
  const btn = document.getElementById('btn_predict_cvd');
  if (!btn) return;
  btn.classList.add('loading');
  btn.querySelector('.btn-spinner')?.classList.add('spinning');
  btn.disabled = true;

  try {
    const data = collectForm('cvd_form');
    const fromSession = document.getElementById('from_session')?.value === '1';
    if (!fromSession) {
      const bmi = calcBMI(data.height_cm, data.weight_kg);
      if (!bmi) { showError('cvd_error', 'Please enter valid height and weight.'); return; }
    }

    clearError('cvd_error');
    const res    = await fetch('/api/predict/cvd', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    const result = await res.json();
    if (!result.success) throw new Error(result.error);

    showCVDResult(result);

  } catch (e) {
    showError('cvd_error', e.message);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-spinner')?.classList.remove('spinning');
    btn.disabled = false;
  }
}

function showCVDResult(r) {
  document.getElementById('cvd_placeholder').style.display = 'none';
  const content = document.getElementById('cvd_result');
  if (content) content.style.display = 'block';

  animateGauge('cvd_gauge_arc', 'cvd_gauge_text', r.prob_pct, 100, r.risk.color);

  const badge = document.getElementById('cvd_risk_badge');
  if (badge) { badge.textContent = r.risk.label; badge.style.color = r.risk.color; }

  const bmiVal = document.getElementById('cvd_bmi_val');
  const bmiCat = document.getElementById('cvd_bmi_cat');
  if (bmiVal) { bmiVal.textContent = r.bmi?.toFixed(1) ?? '—'; bmiVal.style.color = r.bmi_color; }
  if (bmiCat) { bmiCat.textContent = r.bmi_category ?? '';     bmiCat.style.color = r.bmi_color; }

  const diabUsed = document.getElementById('cvd_diabetes_used');
  if (diabUsed) diabUsed.textContent = r.diabetes_used || '—';

  renderSHAPWaterfall('cvd_shap_waterfall', r.shap_drivers);
  const summaryEl = document.getElementById('cvd_shap_summary');
  if (summaryEl) summaryEl.innerHTML = buildSHAPSummary(r.shap_drivers, 'CVD probability');

  const nxt = document.getElementById('cvd_next_btn');
  if (nxt) nxt.style.display = 'flex';

  if (window.innerWidth < 900) content.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('d_height')) onBodyChange('d');
  if (document.getElementById('c_height')) onBodyChange('c');
});
