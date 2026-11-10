const banner = document.querySelector("#banner");
const generateForm = document.querySelector("#generate-form");
const generateButton = document.querySelector("#generate-button");
const copyButton = document.querySelector("#copy-button");
const promptInput = document.querySelector("#prompt");
const maxNewTokensInput = document.querySelector("#max-new-tokens");
const temperatureInput = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperature-value");
const generateNote = document.querySelector("#generate-note");
const output = document.querySelector("#output");

const modelLoaded = document.querySelector("#model-loaded");
const checkpointState = document.querySelector("#checkpoint-state");
const architecture = document.querySelector("#architecture");
const parameterCount = document.querySelector("#parameter-count");
const trainingStatus = document.querySelector("#training-status");
const epochText = document.querySelector("#epoch");
const stepText = document.querySelector("#step");
const trainingLoss = document.querySelector("#training-loss");
const validationLoss = document.querySelector("#validation-loss");
const progressBar = document.querySelector("#progress-bar");
const trainingMessage = document.querySelector("#training-message");
const trainingError = document.querySelector("#training-error");
const trainForm = document.querySelector("#train-form");
const trainButton = document.querySelector("#train-button");
const modeInput = document.querySelector("#mode");
const modeHelp = document.querySelector("#mode-help");

let modelInfo = null;
let currentStatus = null;
let generatedText = "";

function showBanner(message) {
  if (!message) {
    banner.hidden = true;
    banner.textContent = "";
    return;
  }

  banner.hidden = false;
  banner.textContent = message;
}

function formatLoss(value) {
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}

function formatCount(value) {
  return Number.isFinite(value) ? value.toLocaleString("en-US") : "—";
}

async function readJson(response) {
  const text = await response.text();

  if (!text) {
    return {};
  }

  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error("The server returned a response that was not JSON.");
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const body = await readJson(response);

  if (!response.ok) {
    throw new Error(body.error || `Request failed (${response.status}).`);
  }

  return body;
}

function renderModel(info) {
  modelInfo = info;
  const loaded = info.loaded === true;

  modelLoaded.textContent = loaded ? "Loaded" : "Not loaded";
  modelLoaded.className = loaded ? "ok" : "warn";

  if (info.checkpoint && info.checkpoint.valid) {
    checkpointState.textContent = "Available";
    checkpointState.className = "ok";
  } else if (info.checkpoint && info.checkpoint.exists) {
    checkpointState.textContent = "Invalid";
    checkpointState.className = "warn";
  } else {
    checkpointState.textContent = "Missing";
    checkpointState.className = "warn";
  }

  if (loaded) {
    const layerWord = info.numLayers === 1 ? "layer" : "layers";
    architecture.textContent =
      `${info.numLayers} ${layerWord} · ${info.dModel} dimensions · ${info.numHeads} heads · context ${info.maxSequenceLength} · vocab ${info.vocabSize}`;
    parameterCount.textContent = formatCount(info.parameterCount);
  } else {
    architecture.textContent = info.checkpoint && info.checkpoint.error
      ? info.checkpoint.error
      : "No weights are loaded. Fresh training uses the config.js model.";
    parameterCount.textContent = "—";
  }

  generateButton.disabled = !loaded || generateButton.dataset.busy === "true";
  generateNote.textContent = loaded
    ? ""
    : "Generate stays disabled until a trained checkpoint is loaded.";

  updateModeHelp();
  updateTrainButton();
}

function updateModeHelp() {
  if (!modelInfo) {
    modeHelp.textContent = "";
    return;
  }

  if (modeInput.value === "resume") {
    modeHelp.textContent = modelInfo.loaded
      ? "Resume continues from the saved weights. Plain SGD has no saved moments. For this small model, a learning rate around 0.01 is what actually brings the loss down. 0.0003 learns so slowly that the text stays random."
      : "Resume needs a valid checkpoint. Train a fresh model first.";
    return;
  }

  const fresh = modelInfo.freshTraining;
  modeHelp.textContent =
    `Fresh training builds a new ${fresh.numLayers}-layer model ` +
    `(${fresh.dModel} dimensions, context ${fresh.maxSequenceLength}, ` +
    `${formatCount(fresh.parameterCount)} parameters, batch ${fresh.batchSize}). ` +
    "The checkpoint is replaced only after training succeeds.";
}

function renderStatus(status) {
  currentStatus = status;
  const running = status.status === "running";

  trainingStatus.textContent = status.status;
  epochText.textContent = status.totalEpochs
    ? `${status.epoch} / ${status.totalEpochs}`
    : "—";
  stepText.textContent = status.totalSteps
    ? `${status.step} / ${status.totalSteps}`
    : "—";
  trainingLoss.textContent = formatLoss(status.trainingLoss);
  validationLoss.textContent = formatLoss(status.validationLoss);
  trainingMessage.textContent = status.message || "";

  const planned = status.plannedSteps || 0;
  const completed = status.completedSteps || 0;
  const percent = planned > 0 ? Math.min(100, (completed / planned) * 100) : 0;
  progressBar.style.width = `${percent}%`;

  if (status.error) {
    trainingError.hidden = false;
    trainingError.textContent = status.error;
  } else {
    trainingError.hidden = true;
    trainingError.textContent = "";
  }

  trainButton.disabled = running;
  trainButton.textContent = running ? "Training…" : "Start training";
  modeInput.disabled = running;
}

function updateTrainButton() {
  if (!currentStatus || currentStatus.status !== "running") {
    trainButton.disabled = false;
    trainButton.textContent = "Start training";
  }
}

function connectEvents() {
  const source = new EventSource("/api/train/events");

  const apply = (event) => {
    renderStatus(JSON.parse(event.data));
  };

  source.addEventListener("progress", apply);
  source.addEventListener("complete", async (event) => {
    apply(event);
    try {
      renderModel(await fetchJson("/api/model"));
    } catch (error) {
      showBanner(error.message);
    }
  });
  source.addEventListener("failed", apply);
  source.onerror = () => {
    // The browser reconnects. Training continues on the server.
  };
}

temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(1);
});

modeInput.addEventListener("change", updateModeHelp);

generateForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showBanner("");
  generateButton.dataset.busy = "true";
  generateButton.disabled = true;
  generateButton.textContent = "Generating…";

  try {
    const body = await fetchJson("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptInput.value,
        maxNewTokens: Number(maxNewTokensInput.value),
        temperature: Number(temperatureInput.value)
      })
    });

    generatedText = body.text;
    output.textContent = body.text;
    copyButton.disabled = body.text.length === 0;
    generateNote.textContent = body.note || "";
  } catch (error) {
    showBanner(error.message);
  } finally {
    generateButton.dataset.busy = "false";
    generateButton.disabled = !modelInfo || !modelInfo.loaded;
    generateButton.textContent = "Generate";
  }
});

async function copyGeneratedText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (error) {
      // Some browsers reject the async clipboard API. Try a selection copy.
    }
  }

  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.focus();
  area.select();

  let copied = false;

  try {
    copied = document.execCommand("copy");
  } finally {
    document.body.removeChild(area);
  }

  if (!copied) {
    throw new Error("clipboard blocked");
  }
}

copyButton.addEventListener("click", async () => {
  try {
    await copyGeneratedText(generatedText);
    showBanner("");
    copyButton.textContent = "Copied";
    setTimeout(() => {
      copyButton.textContent = "Copy text";
    }, 1200);
  } catch (error) {
    showBanner("The browser blocked clipboard access.");
  }
});

trainForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  showBanner("");

  const epochs = Number(document.querySelector("#epochs").value);
  const stepsPerEpoch = Number(document.querySelector("#steps-per-epoch").value);
  const learningRate = Number(document.querySelector("#learning-rate").value);

  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 100) {
    showBanner("Epochs must be a whole number from 1 to 100.");
    return;
  }

  if (!Number.isInteger(stepsPerEpoch) || stepsPerEpoch < 1 || stepsPerEpoch > 2000) {
    showBanner("Steps per epoch must be a whole number from 1 to 2000.");
    return;
  }

  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
    showBanner("Learning rate must be greater than 0 and at most 1.");
    return;
  }

  try {
    const status = await fetchJson("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode: modeInput.value,
        epochs,
        stepsPerEpoch,
        learningRate
      })
    });

    renderStatus(status);
  } catch (error) {
    showBanner(error.message);
  }
});

async function boot() {
  try {
    const [info, status] = await Promise.all([
      fetchJson("/api/model"),
      fetchJson("/api/train/status")
    ]);

    renderModel(info);
    renderStatus(status);
    connectEvents();
  } catch (error) {
    showBanner(error.message);
  }
}

boot();
