const form = document.querySelector("#predictionForm");
const resetButton = document.querySelector("#resetButton");
const probabilityValue = document.querySelector("#probabilityValue");
const riskLabel = document.querySelector("#riskLabel");
const riskSummary = document.querySelector("#riskSummary");
const gaugeRing = document.querySelector("#gaugeRing");
const signalList = document.querySelector("#signalList");
const canvas = document.querySelector("#ecgCanvas");
const ctx = canvas.getContext("2d");

const defaultProfile = window.SAMPLE_PROFILES.balanced;

function setFormValues(profile) {
  Object.entries(profile).forEach(([name, value]) => {
    const field = form.elements[name];
    if (!field) return;

    if (field instanceof RadioNodeList) {
      const radio = [...field].find((item) => item.value === String(value));
      if (radio) radio.checked = true;
      return;
    }

    field.value = value;
    const range = form.querySelector(`[data-range-for="${name}"]`);
    if (range) range.value = value;
  });
}

function collectFormValues() {
  return {
    age: Number(form.elements.age.value),
    sex: Number(form.elements.sex.value),
    cp: Number(form.elements.cp.value),
    trestbps: Number(form.elements.trestbps.value),
    chol: Number(form.elements.chol.value),
    fbs: Number(form.elements.fbs.value),
    restecg: Number(form.elements.restecg.value),
    thalach: Number(form.elements.thalach.value),
    exang: Number(form.elements.exang.value),
    oldpeak: Number(form.elements.oldpeak.value),
    slope: Number(form.elements.slope.value),
    ca: Number(form.elements.ca.value),
    thal: Number(form.elements.thal.value),
  };
}

function syncRangeInputs() {
  form.querySelectorAll("[data-range-for]").forEach((range) => {
    const number = form.elements[range.dataset.rangeFor];
    range.addEventListener("input", () => {
      number.value = range.value;
    });
    number.addEventListener("input", () => {
      range.value = number.value;
    });
  });
}

function toneColor(tone) {
  if (tone === "high") return "#d84f35";
  if (tone === "medium") return "#d99721";
  return "#10836f";
}

function updateResult(data) {
  const probability = data.probability ?? (data.prediction === 1 ? 1 : 0);
  const percent = Math.round(probability * 100);
  const color = toneColor(data.risk.tone);

  probabilityValue.textContent = `${percent}%`;
  riskLabel.textContent = data.risk.label;
  riskSummary.textContent = data.risk.summary;
  gaugeRing.style.setProperty("--score", `${percent}%`);
  gaugeRing.style.setProperty("--risk-color", color);
  gaugeRing.dataset.tone = data.risk.tone;

  signalList.innerHTML = "";
  data.signals.forEach((signal) => {
    const item = document.createElement("div");
    item.className = "signal-item";
    item.innerHTML = `<span>${signal.label}</span><strong>${signal.value}</strong>`;
    signalList.appendChild(item);
  });

  drawWave(probability, color);
}

async function predict() {
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(collectFormValues()),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }
  updateResult(data);
}

function drawWave(probability = 0.5, color = "#10836f") {
  const width = canvas.width;
  const height = canvas.height;
  const center = height * 0.52;
  const amplitude = 9 + probability * 20;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#dbe3ea";
  ctx.lineWidth = 1;
  for (let x = 0; x < width; x += 24) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += 24) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();

  for (let x = 0; x <= width; x += 3) {
    const cycle = x % 120;
    let y = center + Math.sin(x / 18) * 3;

    if (cycle > 20 && cycle < 34) y -= amplitude * ((cycle - 20) / 14);
    if (cycle >= 34 && cycle < 44) y += amplitude * 1.9 * ((cycle - 34) / 10) - amplitude;
    if (cycle >= 44 && cycle < 58) y -= amplitude * 0.9 * ((cycle - 44) / 14) - amplitude * 0.9;
    if (cycle >= 78 && cycle < 105) y -= Math.sin(((cycle - 78) / 27) * Math.PI) * amplitude * 0.35;

    if (x === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function showError(message) {
  riskLabel.textContent = "Prediction unavailable";
  riskSummary.textContent = message;
  probabilityValue.textContent = "--";
  gaugeRing.style.setProperty("--score", "0%");
  gaugeRing.style.setProperty("--risk-color", "#64748b");
}

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", async () => {
    setFormValues(window.SAMPLE_PROFILES[button.dataset.sample]);
    try {
      await predict();
    } catch (error) {
      showError(error.message);
    }
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  form.classList.add("is-loading");
  try {
    await predict();
  } catch (error) {
    showError(error.message);
  } finally {
    form.classList.remove("is-loading");
  }
});

resetButton.addEventListener("click", async () => {
  setFormValues(defaultProfile);
  try {
    await predict();
  } catch (error) {
    showError(error.message);
  }
});

syncRangeInputs();
setFormValues(defaultProfile);
drawWave();
predict().catch((error) => showError(error.message));
