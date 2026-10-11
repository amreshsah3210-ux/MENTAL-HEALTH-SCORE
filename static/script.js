// When the page is opened straight from disk (file://) talk to the local API.
// When FastAPI serves the page, use relative URLs.
const API = location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const ARC_LENGTH = 314.16;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (sel) => document.querySelector(sel);
const form = $("#form");

/* ---------- sliders: live value + filled track ---------- */
document.querySelectorAll(".slider").forEach((slider) => {
  const out = document.getElementById(slider.id + "-out");
  const unit = slider.dataset.unit || "";
  const dec = Number(slider.dataset.dec || 0);

  const update = () => {
    const min = Number(slider.min), max = Number(slider.max), val = Number(slider.value);
    slider.style.setProperty("--pct", ((val - min) / (max - min)) * 100 + "%");
    out.textContent = val.toFixed(dec) + unit;
  };
  slider.addEventListener("input", update);
  update();
});

/* ---------- server status ---------- */
async function checkServer() {
  const box = $("#status");
  const text = $("#status-text");
  try {
    const res = await fetch(API + "/api/health");
    const data = await res.json();
    if (data.status === "ok") {
      box.className = "status ok";
      text.textContent = "Model ready";
    } else {
      box.className = "status bad";
      text.textContent = "Model not loaded";
    }
  } catch {
    box.className = "status bad";
    text.textContent = "Server offline";
  }
}
checkServer();

/* ---------- predict ---------- */
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();

  const raw = Object.fromEntries(new FormData(form).entries());
  const payload = {
    age: parseInt(raw.age, 10),
    gender: raw.gender,
    country: raw.country,
    academic_level: raw.academic_level,
    platform: raw.platform,
    purpose: raw.purpose,
    daily_usage_hours: parseFloat(raw.daily_usage_hours),
    daily_unlocks: parseInt(raw.daily_unlocks, 10),
    study_hours: parseFloat(raw.study_hours),
    physical_activity_hours: parseFloat(raw.physical_activity_hours),
    sleep_hours: parseFloat(raw.sleep_hours),
    stress_level: raw.stress_level,
  };

  const button = $("#go");
  button.disabled = true;
  button.textContent = "Predicting…";

  try {
    const res = await fetch(API + "/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));

    if (!res.ok) throw new Error(readError(data, res.status));

    showResult(data);
    checkServer();
    if (window.matchMedia("(max-width: 960px)").matches) {
      $("#result").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
    }
  } catch (err) {
    const offline = err instanceof TypeError;
    showError(
      offline
        ? "Can't reach the server. Run start_app.bat from the project folder."
        : err.message
    );
    if (offline) checkServer();
  } finally {
    button.disabled = false;
    button.textContent = "Predict my score";
  }
});

function readError(data, status) {
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail) && data.detail.length) {
    const first = data.detail[0];
    const field = (first.loc || []).slice(-1)[0];
    return `Check the "${String(field).replace(/_/g, " ")}" value: ${first.msg}.`;
  }
  return `Something went wrong (status ${status}). Try again.`;
}

function showError(msg) {
  const el = $("#error");
  el.textContent = msg;
  el.hidden = false;
}
function hideError() {
  $("#error").hidden = true;
}

/* ---------- render result ---------- */
function showResult(d) {
  $("#empty").hidden = true;
  $("#out").hidden = false;

  // gauge arc
  $("#arc").style.strokeDashoffset = ARC_LENGTH * (1 - Math.min(d.score, 10) / 10);
  countUp($("#score"), d.score);

  // band + message
  const band = $("#band");
  band.textContent = d.band;
  band.className = d.band.toLowerCase();
  $("#message").textContent = d.message;

  // dataset comparison
  const pct = $("#percentile");
  const avg = $("#average");
  pct.textContent = d.percentile != null ? `Higher than ${d.percentile.toFixed(0)}% of students` : "Not available";
  avg.textContent = d.dataset_average != null ? d.dataset_average.toFixed(2) : "Not available";

  // what-if
  const list = $("#whatif");
  list.innerHTML = "";
  $("#whatif-wrap").hidden = d.what_if.length === 0;

  d.what_if.forEach((w) => {
    const up = w.change >= 0;
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="row">
        <span></span>
        <span class="delta ${up ? "up" : "down"}">${up ? "+" : ""}${w.change.toFixed(2)}</span>
      </div>
      <div class="track"><div class="fill ${up ? "" : "neg"}"></div></div>`;
    li.querySelector(".row span").textContent = w.label; // textContent keeps it safe
    list.appendChild(li);

    const width = Math.min(Math.abs(w.change) / 1, 1) * 100; // 1.0 point fills the bar
    requestAnimationFrame(() => {
      li.querySelector(".fill").style.width = Math.max(width, 3) + "%";
    });
  });
}

function countUp(el, target) {
  if (reduceMotion) {
    el.textContent = target.toFixed(1);
    return;
  }
  const start = performance.now();
  const duration = 1000;
  const from = parseFloat(el.textContent) || 0;

  function frame(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (target - from) * eased).toFixed(1);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
