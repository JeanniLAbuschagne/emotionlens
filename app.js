// app.js — client-side logic for EmotionLens
// Handles file selection, draws the image (with the detected face box) on a
// canvas, calls POST /predict, and renders the emotion + probability bars.

const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const dropText = document.getElementById("dropText");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultEl = document.getElementById("result");
const emotionBig = document.getElementById("emotionBig");
const valenceEl = document.getElementById("valence");
const confidenceEl = document.getElementById("confidence");
const barsEl = document.getElementById("bars");
const warnEl = document.getElementById("warn");
const statusEl = document.getElementById("status");

// colour per valence so the UI reads at a glance
const VALENCE_COLOR = { positive: "#2dd4a7", neutral: "#9aa6c4", negative: "#f0606b" };

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

// drag-and-drop support
["dragover", "dragenter"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

async function handleFile(file) {
  if (!file) return;
  dropText.textContent = file.name;
  await drawImage(file);

  statusEl.textContent = "Analysing…";
  resultEl.hidden = true;

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }
    const data = await res.json();
    render(data);
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
  }
}

// draw the uploaded image onto the canvas, then overlay the face box if any
function drawImage(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const maxW = 360;
      const scale = Math.min(1, maxW / img.width);
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas._scale = scale; // remember for box overlay
      resolve();
    };
    img.src = URL.createObjectURL(file);
  });
}

function render(data) {
  resultEl.hidden = false;

  emotionBig.textContent = data.emotion;
  emotionBig.style.color = VALENCE_COLOR[data.valence] || "#fff";
  valenceEl.textContent = "valence: " + data.valence;
  confidenceEl.textContent = "confidence: " + (data.confidence * 100).toFixed(1) + "%";
  warnEl.hidden = !data.uncertain;

  // probability bars, sorted high -> low
  barsEl.innerHTML = "";
  Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, p]) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        `<span class="bar-label">${label}</span>` +
        `<span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span></span>` +
        `<span class="bar-val">${(p * 100).toFixed(0)}%</span>`;
      barsEl.appendChild(row);
    });

  // overlay the detected face box on the canvas
  if (data.box && canvas._scale) {
    const [x, y, w, h] = data.box;
    const s = canvas._scale;
    ctx.strokeStyle = VALENCE_COLOR[data.valence] || "#fff";
    ctx.lineWidth = 3;
    ctx.strokeRect(x * s, y * s, w * s, h * s);
  }
}
