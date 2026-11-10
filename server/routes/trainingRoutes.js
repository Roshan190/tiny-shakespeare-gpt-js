const express = require("express");

const trainingService = require("../services/trainingService");
const { httpError } = require("../middleware/errorHandler");

const router = express.Router();

function rejectUnknownFields(body, allowed) {
  for (const key of Object.keys(body)) {
    if (!allowed.includes(key)) {
      throw httpError(400, `Unknown field "${key}".`);
    }
  }
}

function parseTrainRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw httpError(400, "Request body must be a JSON object.");
  }

  rejectUnknownFields(body, [
    "mode",
    "epochs",
    "stepsPerEpoch",
    "learningRate"
  ]);

  const { mode, epochs, stepsPerEpoch, learningRate } = body;

  if (mode !== "fresh" && mode !== "resume") {
    throw httpError(400, 'mode must be "fresh" or "resume".');
  }

  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 100) {
    throw httpError(400, "epochs must be an integer from 1 to 100.");
  }

  if (!Number.isInteger(stepsPerEpoch) || stepsPerEpoch < 1 || stepsPerEpoch > 2000) {
    throw httpError(400, "stepsPerEpoch must be an integer from 1 to 2000.");
  }

  if (
    typeof learningRate !== "number" ||
    !Number.isFinite(learningRate) ||
    learningRate <= 0 ||
    learningRate > 1
  ) {
    throw httpError(400, "learningRate must be a number greater than 0 and at most 1.");
  }

  return { mode, epochs, stepsPerEpoch, learningRate };
}

router.post("/train", (req, res, next) => {
  try {
    const request = parseTrainRequest(req.body);
    const job = trainingService.startTraining(request);
    res.status(202).json(job);
  } catch (error) {
    next(error);
  }
});

router.get("/train/status", (req, res) => {
  res.json(trainingService.getStatus());
});

router.get("/train/events", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();
  trainingService.addClient(res);
});

module.exports = router;
