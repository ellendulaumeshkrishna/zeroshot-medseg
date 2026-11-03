// zsmedseg demo page. Plain JavaScript, no build step.

const $ = (id) => document.getElementById(id);
const state = { image: null, mask: null, presets: [], ready: false, last: null };

// ---------------------------------------------------------------- server status
async function pollStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const pill = $("status");
    if (s.error) {
      pill.className = "status failed";
      $("status-text").textContent = "Model loading failed";
      showError("The models could not be loaded: " + s.error);
      return;
    }
    state.ready = s.ready;
    pill.className = "status" + (s.ready ? " ready" : "");
    $("status-text").textContent = s.ready ? `Ready · ${s.device}` : "Loading models…";
    updateRunButton();
    if (!s.ready) setTimeout(pollStatus, 2000);
  } catch {
    $("status-text").textContent = "Server not reachable";
    setTimeout(pollStatus, 3000);
  }
}

function updateRunButton() {
  $("run").disabled = !(state.ready && state.image && $("target").value.trim());
  $("run").textContent = state.ready ? "Segment" : "Waiting for the models…";
}

// ---------------------------------------------------------------- presets and examples
async function loadPresets() {
  state.presets = await (await fetch("/api/presets")).json();
  const select = $("preset");
  for (const p of state.presets) select.add(new Option(p.label, p.key));
  select.add(new Option("Custom: type your own description", "custom"));
  select.addEventListener("change", () => applyPreset(select.value));
  applyPreset(state.presets[0].key);

  const box = $("examples");
  for (const p of state.presets.filter((p) => p.example)) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "example";
    b.dataset.key = p.key;
    b.innerHTML = `<img src="${p.example}" alt=""><span>${p.label.split(",")[0]}</span>`;
    b.addEventListener("click", () => useExample(p));
    box.appendChild(b);
  }
}

function applyPreset(key) {
  const p = state.presets.find((x) => x.key === key);
  if (p) {
    $("target").value = p.target.join(", ");
    $("background").value = p.background.join(", ");
    document.querySelector(`input[name=objects][value="${p.num_objects}"]`).checked = true;
  }
  $("preset").value = key;
  updateRunButton();
}

async function useExample(p) {
  const get = (url) => fetch(url, { cache: "no-cache" }).then((r) => r.blob());
  const [img, mask] = await Promise.all([get(p.example), get(p.example_mask)]);
  setImage(new File([img], `${p.key}.png`, { type: "image/png" }));
  state.mask = new File([mask], `${p.key}_mask.png`, { type: "image/png" });
  $("mask").value = "";
  $("mask-note").textContent = "Using the expert mask of this test image.";
  applyPreset(p.key);
  document.querySelectorAll(".example").forEach((e) => e.classList.toggle("active", e.dataset.key === p.key));
}

// ---------------------------------------------------------------- file inputs
function setImage(file) {
  state.image = file;
  const url = URL.createObjectURL(file);
  $("preview").src = url;
  $("preview").hidden = false;
  $("drop-text").hidden = true;
  $("drop").classList.add("filled");
  updateRunButton();
}

function clearExampleSelection() {
  state.mask = null;
  $("mask-note").textContent = "Add the expert mask to see the Dice score.";
  document.querySelectorAll(".example").forEach((e) => e.classList.remove("active"));
}

$("image").addEventListener("change", (e) => {
  if (e.target.files[0]) { clearExampleSelection(); setImage(e.target.files[0]); }
});
$("mask").addEventListener("change", (e) => {
  state.mask = e.target.files[0] || null;
  $("mask-note").textContent = state.mask ? "The Dice score will be shown." : "Add the expert mask to see the Dice score.";
});
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) { clearExampleSelection(); setImage(f); }
});
$("target").addEventListener("input", () => { $("preset").value = "custom"; updateRunButton(); });
$("background").addEventListener("input", () => { $("preset").value = "custom"; });

// ---------------------------------------------------------------- run
$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if ($("run").disabled) return;
  const form = new FormData();
  form.append("image", state.image);
  if (state.mask) form.append("mask", state.mask);
  form.append("target", $("target").value);
  form.append("background", $("background").value);
  form.append("num_objects", document.querySelector("input[name=objects]:checked").value);

  show("busy");
  $("run").disabled = true;
  try {
    const r = await fetch("/api/segment", { method: "POST", body: form });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    render(data);
  } catch (err) {
    show("empty");
    showError(err.message);
  } finally {
    updateRunButton();
  }
});

function show(which) {
  for (const id of ["empty", "busy", "output"]) $(id).hidden = id !== which;
  $("error").hidden = true;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = false;
}

// ---------------------------------------------------------------- results
function render(d) {
  state.last = d;
  const m = d.metrics;
  const items = [];
  if (m.dice !== undefined) {
    items.push(["Dice", m.dice.toFixed(2), "headline"]);
    items.push(["IoU", m.iou.toFixed(2), ""]);
  }
  items.push(["MedSAM2 predicted IoU", m.predicted_iou.toFixed(3), m.dice === undefined ? "headline" : ""]);
  items.push(["Time", `${m.seconds}<small> s</small>`, ""]);
  items.push(["Image", `${d.size[0]}×${d.size[1]}`, ""]);
  $("metrics").innerHTML = items.map(([k, v, c]) => `<div class="${c}"><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  $("img-input").src = d.images.input;
  $("img-saliency").src = d.images.saliency;
  const hasGt = Boolean(d.images.final_with_gt);
  $("gt-toggle").hidden = !hasGt;
  $("legend-gt").hidden = $("legend-gt-text").hidden = !hasGt;
  setFinalImage();
  $("dl-mask").href = d.images.mask;
  $("dl-overlay").href = d.images.final;

  // Candidates: one column per point, one row per MedSAM2 mask.
  const nPoints = d.points.length;
  const box = $("candidates");
  box.style.gridTemplateColumns = `repeat(${Math.max(nPoints, 1)}, minmax(0, 1fr))`;
  let html = "";
  for (let p = 1; p <= nPoints; p++) html += `<div class="col-head">Point ${p}</div>`;
  for (let k = 1; k <= 3; k++) {
    for (let p = 1; p <= nPoints; p++) {
      const c = d.candidates.find((x) => x.point === p && x.mask === k);
      html += c
        ? `<div class="cand${c.kept ? " kept" : ""}"><img src="${c.image}" alt="Candidate ${k} of point ${p}" class="zoomable">
           <div><span>${c.kept ? "<b>kept</b>" : "&nbsp;"}</span><b>${c.score.toFixed(3)}</b></div></div>`
        : "<div></div>";
    }
  }
  box.innerHTML = html;
  $("kept-note").textContent = d.num_kept > 1
    ? `(numbers: predicted IoU; the ${d.num_kept} kept masks are joined into one)`
    : "(numbers: predicted IoU; the highest one is kept)";

  const note = $("resize-note");
  note.hidden = !d.resized_from;
  if (d.resized_from) note.textContent =
    `The image was ${d.resized_from[0]}×${d.resized_from[1]} and was shrunk to ${d.size[0]}×${d.size[1]} to fit GPU memory.`;
  show("output");
}

function setFinalImage() {
  const d = state.last;
  if (!d) return;
  $("img-final").src = $("show-gt").checked && d.images.final_with_gt ? d.images.final_with_gt : d.images.final;
}
$("show-gt").addEventListener("change", setFinalImage);

// ---------------------------------------------------------------- zoom
document.addEventListener("click", (e) => {
  if (e.target.classList?.contains("zoomable")) {
    $("lightbox").querySelector("img").src = e.target.src;
    $("lightbox").hidden = false;
  }
});
$("lightbox").addEventListener("click", () => { $("lightbox").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("lightbox").hidden = true; });

loadPresets().then(pollStatus);
