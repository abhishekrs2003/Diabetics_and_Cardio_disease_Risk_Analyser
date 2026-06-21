/* ============================================================
   CardioRisk AI — Shared Utilities (utils.js)
   ============================================================ */

function calcBMI(heightCm, weightKg) {
  const h = parseFloat(heightCm), w = parseFloat(weightKg);
  if (!h || !w || h <= 0 || w <= 0) return null;
  return Math.round((w / Math.pow(h / 100, 2)) * 10) / 10;
}

function bmiMeta(bmi) {
  if (bmi === null) return { cat: 'Enter height & weight', color: '#4e5a6b' };
  if (bmi < 18.5)  return { cat: 'Underweight',   color: '#60a5fa' };
  if (bmi < 25)    return { cat: 'Normal weight',  color: '#22c55e' };
  if (bmi < 30)    return { cat: 'Overweight',     color: '#f59e0b' };
  if (bmi < 35)    return { cat: 'Obese (Class I)',color: '#ef4444' };
  if (bmi < 40)    return { cat: 'Obese (Class II)',color:'#dc2626' };
  return               { cat: 'Obese (Class III)', color: '#7f1d1d' };
}

function bmiColor(bmi) {
  if (!bmi)       return '#94a3b8';
  if (bmi < 18.5) return '#60a5fa';
  if (bmi < 25)   return '#22c55e';
  if (bmi < 30)   return '#f59e0b';
  if (bmi < 35)   return '#ef4444';
  if (bmi < 40)   return '#dc2626';
  return '#7f1d1d';
}

function updateBMICard(heightId, weightId, valId, catId, markerId) {
  const h   = document.getElementById(heightId)?.value;
  const w   = document.getElementById(weightId)?.value;
  const bmi = calcBMI(h, w);
  const meta= bmiMeta(bmi);

  const valEl  = document.getElementById(valId);
  const catEl  = document.getElementById(catId);
  const markEl = document.getElementById(markerId);

  if (valEl) { valEl.textContent = bmi ? bmi.toFixed(1) : '—'; valEl.style.color = meta.color; }
  if (catEl) { catEl.textContent = meta.cat; catEl.style.color = meta.color; }
  if (markEl) {
    const pct = bmi ? Math.min(100, Math.max(0, (bmi - 15) / 25 * 100)) : 0;
    markEl.style.left = pct + '%';
    markEl.style.background = bmi ? meta.color : '#4e5a6b';
  }
}

function onBodyChange(prefix) {
  updateBMICard(`${prefix}_height`, `${prefix}_weight`,
                `${prefix}_bmi_val`, `${prefix}_bmi_cat`, `${prefix}_bmi_marker`);
}

function syncRangeInline(inputEl, spanId, suffix) {
  const el = document.getElementById(spanId);
  if (!el) return;
  const raw = parseFloat(inputEl.value);
  const decimals = (inputEl.step && parseFloat(inputEl.step) < 1) ? 1 : 0;
  el.textContent = raw.toFixed(decimals) + (suffix || '');
}

function syncRange(inputEl, displayId) {
  const el = document.getElementById(displayId);
  if (el) el.textContent = parseFloat(inputEl.value).toFixed(
    inputEl.step && parseFloat(inputEl.step) < 1 ? 1 : 0
  );
}

function collectForm(formId) {
  const form = document.getElementById(formId);
  if (!form) return {};
  const data = {};
  new FormData(form).forEach((v, k) => { data[k] = v; });
  form.querySelectorAll('input[type="range"]').forEach(el => {
    if (el.name) data[el.name] = el.value;
  });
  return data;
}

function animateGauge(arcId, textId, value, maxVal, color) {
  const arc  = document.getElementById(arcId);
  const text = document.getElementById(textId);
  if (!arc || !text) return;
  const total  = 251.2;
  const pct    = Math.min(Math.max(value / maxVal, 0), 1);
  arc.style.strokeDashoffset = (total - pct * total).toString();
  arc.style.stroke           = color;
  let cur = 0;
  const target  = Math.round(value * 10) / 10;
  const isFloat = maxVal <= 10;
  const step = () => {
    cur = Math.min(cur + maxVal / 55, target);
    text.textContent = isFloat ? cur.toFixed(1) : Math.round(cur);
    if (cur < target) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/* ── SHAP waterfall renderer ──────────────────────────────────
   drivers: [{feature, label, shap_value, direction, magnitude_pct}]
   Renders a two-sided bar chart: red bars push right (increase risk),
   green bars push left (decrease risk) — centred at zero.
*/
function renderSHAPWaterfall(containerId, drivers) {
  const el = document.getElementById(containerId);
  if (!el) return;

  if (!drivers || !drivers.length) {
    el.innerHTML = '<p style="font-size:.78rem;color:var(--text2)">No SHAP data available.</p>';
    return;
  }

  el.innerHTML = drivers.map((d, i) => {
    const isIncrease = d.direction === 'increases';
    const icon = isIncrease ? '▲' : '▼';
    const iconColor = isIncrease ? '#ef4444' : '#22c55e';
    return `<div class="shap-driver-row" style="animation-delay:${i * 60}ms">
      <span class="shap-driver-icon" style="color:${iconColor}">${icon}</span>
      <span class="shap-driver-label" title="${d.label}">${d.label}</span>
      <div class="shap-driver-track">
        <div class="shap-driver-center"></div>
        <div class="shap-driver-bar ${isIncrease ? 'increase' : 'decrease'}"
             data-pct="${d.magnitude_pct / 2}"></div>
      </div>
      <span class="shap-driver-val" style="color:${iconColor}">
        ${isIncrease ? '+' : '−'}${Math.abs(d.shap_value).toFixed(3)}
      </span>
    </div>`;
  }).join('');

  setTimeout(() => {
    el.querySelectorAll('.shap-driver-bar').forEach(bar => {
      bar.style.width = bar.dataset.pct + '%';
    });
  }, 80);
}

function renderSHAPBars(containerId, marginals) {
  const el = document.getElementById(containerId);
  if (!el) return;
  const entries = Object.entries(marginals)
    .filter(([, v]) => Math.abs(v.pct) > 0.001)
    .sort((a, b) => Math.abs(b[1].pct) - Math.abs(a[1].pct));
  if (!entries.length) {
    el.innerHTML = '<p style="font-size:.78rem;color:var(--text2)">Adjust the controls to see impact</p>';
    return;
  }
  const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v.pct)), 0.001);
  el.innerHTML = entries.map(([feat, d]) => {
    const pct   = (Math.abs(d.pct) / maxAbs * 100).toFixed(1);
    const color = d.pct < 0 ? '#22c55e' : '#ef4444';
    const sign  = d.pct < 0 ? '−' : '+';
    const name  = feat.replace(/_/g, ' ');
    return `<div class="shap-row">
      <span class="shap-name" title="${name}">${name}</span>
      <div class="shap-track"><div class="shap-fill" style="width:0%;background:${color}" data-pct="${pct}"></div></div>
      <span class="shap-pct" style="color:${color}">${sign}${Math.abs(d.pct).toFixed(1)}%</span>
    </div>`;
  }).join('');
  setTimeout(() => { el.querySelectorAll('.shap-fill').forEach(b => { b.style.width = b.dataset.pct + '%'; }); }, 60);
}

function showError(id, msg) {
  const el = document.getElementById(id);
  if (el) { el.textContent = msg; el.style.display = 'block'; }
  else    { alert(msg); }
}
function clearError(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}

function renderDots(containerId, total, current) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = Array.from({ length: total }, (_, i) =>
    `<span class="pdot ${i === current ? 'active' : i < current ? 'done' : ''}"></span>`
  ).join('');
}

/* ── Wizard controller ────────────────────────────────────────
   Generic step wizard used by both diabetes.html and cvd.html.
*/
function initWizard(tabsContainerSelector, sectionIds, dotContainerId,
                     backBtnId, nextBtnId, predictBtnId) {
  let step = 0;

  function show(i) {
    step = Math.max(0, Math.min(sectionIds.length - 1, i));
    sectionIds.forEach((id, j) => {
      const el = document.getElementById(id);
      if (el) el.classList.toggle('active', j === step);
    });
    document.querySelectorAll(tabsContainerSelector + ' .wtab').forEach((t, j) => {
      t.classList.toggle('active', j === step);
    });
    const back    = document.getElementById(backBtnId);
    const next    = document.getElementById(nextBtnId);
    const predict = document.getElementById(predictBtnId);
    const isLast  = step === sectionIds.length - 1;
    if (back)    back.style.visibility = step === 0 ? 'hidden' : 'visible';
    if (next)    next.style.display    = isLast ? 'none' : '';
    if (predict) predict.classList.toggle('show', isLast);
    renderDots(dotContainerId, sectionIds.length, step);
  }

  document.querySelectorAll(tabsContainerSelector + ' .wtab').forEach((tab, i) => {
    tab.addEventListener('click', () => show(i));
  });

  window._wizardNav = (dir) => show(step + dir);
  show(0);
}

function wizardNav(dir) {
  if (window._wizardNav) window._wizardNav(dir);
}
