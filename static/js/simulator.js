/* ============================================================
   CardioRisk AI — Simulator page JS (simulator.js)
   ============================================================ */

let _simModel   = 'cvd';
let _debounce   = null;
let _baseInputs = {};

function setSimModel(model) {
  _simModel = model;
  document.querySelectorAll('.sim-tab').forEach(t => t.classList.toggle('active', t.dataset.model === model));
  document.getElementById('cvd_controls').style.display  = model === 'cvd'      ? '' : 'none';
  document.getElementById('diab_controls').style.display = model === 'diabetes'  ? '' : 'none';
  runSim();
}

function collectSimInputs() {
  const controls = document.getElementById(_simModel + '_controls');
  if (!controls) return {};
  const inp = {};
  controls.querySelectorAll('input, select').forEach(el => {
    if (!el.name) return;
    if (el.type === 'radio' && !el.checked) return;
    inp[el.name] = el.type === 'range' ? parseFloat(el.value) : el.value;
  });
  return inp;
}

function runSim() {
  clearTimeout(_debounce);
  _debounce = setTimeout(_doSim, 280);
}

async function _doSim() {
  const inputs = { ..._baseInputs, ...collectSimInputs() };
  try {
    const res  = await fetch('/api/simulate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: _simModel, inputs: inputs, interventions: collectSimInputs() }),
    });
    const data = await res.json();
    if (!data.success) return;
    updateSimDisplay(data);
  } catch (e) { console.error('Simulator error:', e); }
}

function updateSimDisplay(data) {
  const baseEl = document.getElementById('sim_baseline');
  const newEl  = document.getElementById('sim_new');
  if (baseEl) baseEl.textContent = _simModel === 'cvd' ? (data.baseline * 100).toFixed(1) + '%' : data.baseline.toFixed(1);
  if (newEl) {
    newEl.textContent = _simModel === 'cvd' ? (data.new_score * 100).toFixed(1) + '%' : data.new_score.toFixed(1);
    newEl.style.color = data.pct_change <= 0 ? '#22c55e' : '#ef4444';
  }

  const dNum = document.getElementById('sim_delta_num');
  const dSub = document.getElementById('sim_delta_sub');
  if (dNum) {
    const pct = data.pct_change;
    dNum.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
    dNum.style.color = pct <= 0 ? '#22c55e' : '#ef4444';
  }
  if (dSub) {
    const pct = data.pct_change;
    dSub.textContent = pct < -0.1
      ? `Risk reduced by ${Math.abs(pct).toFixed(1)}% with these changes`
      : pct > 0.1
        ? `Risk increased by ${pct.toFixed(1)}% — adjust to reduce`
        : 'No significant change from baseline';
  }

  Object.entries(data.marginals || {}).forEach(([feat, d]) => {
    const el = document.getElementById(`imp_${feat}`);
    if (!el) return;
    el.textContent = (d.pct >= 0 ? '+' : '') + d.pct.toFixed(1) + '%';
    el.style.color = d.pct <= 0 ? '#22c55e' : '#ef4444';
  });

  renderSHAPBars('sim_shap_bars', data.marginals || {});
}

const CVD_PRESETS = {
  optimal:  { Smoking_History:'No', Exercise:'Yes', BMI: 22, Alcohol_Consumption: 1, FriedPotato_Consumption: 2, Fruit_Consumption: 25 },
  no_smoke: { Smoking_History: 'No' },
  exercise: { Exercise: 'Yes' },
  bmi:      { BMI: 24 },
  reset:    { Smoking_History:'No', Exercise:'Yes', BMI: 27, Alcohol_Consumption: 4, FriedPotato_Consumption: 10 },
};
const DIAB_PRESETS = {
  optimal:  { physical_activity_level:'High', stress_level: 1, sugar_intake_grams_per_day: 20, sleep_hours: 8 },
  exercise: { physical_activity_level: 'High' },
  sleep:    { sleep_hours: 8 },
  sugar:    { sugar_intake_grams_per_day: 20 },
  reset:    { physical_activity_level:'Low', stress_level: 2, sugar_intake_grams_per_day: 70, sleep_hours: 6 },
};

function applyPreset(name) {
  const presets = _simModel === 'cvd' ? CVD_PRESETS : DIAB_PRESETS;
  const p = presets[name] || {};
  const controls = document.getElementById(_simModel + '_controls');
  if (!controls) return;
  Object.entries(p).forEach(([feat, val]) => {
    const el = controls.querySelector(`[name="${feat}"]`);
    if (!el) return;
    if (el.type === 'range') { el.value = val; el.dispatchEvent(new Event('input')); }
    else if (el.tagName === 'SELECT') { el.value = val; }
    else if (el.type === 'radio') { controls.querySelectorAll(`[name="${feat}"]`).forEach(r => { r.checked = r.value == val; }); }
  });
  runSim();
}

function setBaseInputs(inputs) { _baseInputs = inputs || {}; }

document.addEventListener('DOMContentLoaded', () => {
  const page = document.getElementById('simulator_page');
  if (page?.dataset.model) setSimModel(page.dataset.model);
  else runSim();
});
