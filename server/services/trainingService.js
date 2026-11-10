const path = require("path");
const crypto = require("crypto");
const { Worker } = require("worker_threads");

const modelService = require("./modelService");

const clients = new Set();

const job = {
  status: "idle",
  jobId: null,
  mode: null,
  epoch: 0,
  totalEpochs: 0,
  step: 0,
  totalSteps: 0,
  trainingLoss: null,
  validationLoss: null,
  startedAt: null,
  finishedAt: null,
  error: null,
  message: "No training job has been started."
};

let worker = null;
let runToken = 0;

function isRunning() {
  return job.status === "running";
}

function snapshot() {
  const completedSteps = job.totalSteps > 0 && job.epoch > 0
    ? (job.epoch - 1) * job.totalSteps + job.step
    : 0;

  return {
    ...job,
    completedSteps,
    plannedSteps: job.totalEpochs * job.totalSteps,
    training: job.status === "running"
  };
}

function writeEvent(response, event, data) {
  response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function broadcast(event, data) {
  for (const response of clients) {
    try {
      writeEvent(response, event, data);
    } catch (error) {
      clients.delete(response);
    }
  }
}

function markFailed(token, message) {
  if (token !== runToken || job.status !== "running") {
    return;
  }

  job.status = "failed";
  job.error = message;
  job.finishedAt = new Date().toISOString();
  job.message = "Training failed. The previous checkpoint was left unchanged.";
  broadcast("failed", snapshot());
}

function startTraining({ mode, epochs, stepsPerEpoch, learningRate }) {
  if (isRunning()) {
    const error = new Error("A training job is already running.");
    error.status = 409;
    throw error;
  }

  if (mode === "resume" && !modelService.hasValidCheckpoint()) {
    const reason = modelService.getCheckpointError();
    const error = new Error(
      reason
        ? `Cannot resume training: ${reason}`
        : "Cannot resume training because no valid checkpoint exists."
    );
    error.status = 400;
    throw error;
  }

  const token = ++runToken;

  job.status = "running";
  job.jobId = crypto.randomUUID();
  job.mode = mode;
  job.epoch = 0;
  job.totalEpochs = epochs;
  job.step = 0;
  job.totalSteps = stepsPerEpoch;
  job.trainingLoss = null;
  job.validationLoss = null;
  job.startedAt = new Date().toISOString();
  job.finishedAt = null;
  job.error = null;
  job.message = mode === "resume"
    ? "Training resumed from the saved weights. Optimizer state was reset because SGD does not store moments."
    : "Fresh training started from new random weights.";

  broadcast("progress", snapshot());

  const activeWorker = new Worker(
    path.join(__dirname, "../workers/trainWorker.js"),
    {
      workerData: {
        mode,
        epochs,
        stepsPerEpoch,
        learningRate,
        checkpointPath: modelService.getCheckpointPath()
      }
    }
  );

  worker = activeWorker;
  let settled = false;

  activeWorker.on("message", (message) => {
    if (token !== runToken) {
      return;
    }

    if (message.type === "progress") {
      const progress = message.progress;
      job.epoch = progress.epoch;
      job.totalEpochs = progress.totalEpochs;
      job.step = progress.step;
      job.totalSteps = progress.totalSteps;
      job.trainingLoss = progress.trainingLoss;

      if (Number.isFinite(progress.validationLoss)) {
        job.validationLoss = progress.validationLoss;
      }

      job.message = progress.phase === "epoch"
        ? `Completed epoch ${progress.epoch} of ${progress.totalEpochs}.`
        : `Epoch ${progress.epoch}, step ${progress.step} of ${progress.totalSteps}.`;

      if (progress.phase === "epoch") {
        try {
          modelService.reload();
          job.message = `Completed epoch ${progress.epoch} of ${progress.totalEpochs}. The latest weights are loaded.`;
        } catch (error) {
          console.error("Could not reload the epoch checkpoint:", error.message);
        }
      }

      broadcast("progress", snapshot());
      return;
    }

    if (message.type === "complete") {
      settled = true;

      try {
        modelService.reload();
        job.status = "completed";
        job.finishedAt = new Date().toISOString();
        job.message = "Training completed and the new checkpoint is loaded.";
        broadcast("complete", snapshot());
      } catch (error) {
        job.status = "failed";
        job.error = `Training finished, but the checkpoint could not be loaded: ${error.message}`;
        job.finishedAt = new Date().toISOString();
        job.message = "Training finished, but the server could not load the new checkpoint.";
        broadcast("failed", snapshot());
      }

      return;
    }

    if (message.type === "error") {
      settled = true;
      markFailed(token, message.message || "Training failed.");
    }
  });

  activeWorker.on("error", (error) => {
    settled = true;
    markFailed(token, error.message);
  });

  activeWorker.on("exit", (code) => {
    if (worker === activeWorker) {
      worker = null;
    }

    if (!settled) {
      markFailed(
        token,
        `Training process exited with code ${code}.`
      );
    }
  });

  return snapshot();
}

function addClient(response) {
  clients.add(response);
  writeEvent(response, "progress", snapshot());

  const heartbeat = setInterval(() => {
    response.write(": heartbeat\n\n");
  }, 15000);

  response.on("close", () => {
    clearInterval(heartbeat);
    clients.delete(response);
  });
}

function getStatus() {
  return snapshot();
}

module.exports = {
  startTraining,
  getStatus,
  addClient,
  isRunning
};
