const express = require("express");

const { generateText } = require("../../generation/generate");
const modelService = require("../services/modelService");
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

function parseGenerateRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw httpError(400, "Request body must be a JSON object.");
  }

  rejectUnknownFields(body, ["prompt", "maxNewTokens", "temperature"]);

  const { prompt, maxNewTokens, temperature } = body;

  if (typeof prompt !== "string" || prompt.length === 0) {
    throw httpError(400, "prompt must be a non-empty string.");
  }

  if (prompt.length > 2000) {
    throw httpError(400, "prompt must be 2000 characters or fewer.");
  }

  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0 || maxNewTokens > 1000) {
    throw httpError(400, "maxNewTokens must be an integer from 0 to 1000.");
  }

  if (
    typeof temperature !== "number" ||
    !Number.isFinite(temperature) ||
    temperature <= 0 ||
    temperature > 5
  ) {
    throw httpError(400, "temperature must be a number greater than 0 and at most 5.");
  }

  return { prompt, maxNewTokens, temperature };
}

router.get("/health", (req, res) => {
  const runtime = modelService.getRuntimeState();

  res.json({
    status: "ok",
    modelLoaded: runtime.modelLoaded,
    checkpointExists: runtime.checkpointExists,
    checkpointValid: runtime.checkpointValid,
    checkpointError: runtime.checkpointError,
    training: trainingService.isRunning()
  });
});

router.get("/model", (req, res) => {
  res.json(modelService.getModelInfo());
});

router.post("/generate", (req, res, next) => {
  try {
    const input = parseGenerateRequest(req.body);

    if (!modelService.isReady()) {
      throw httpError(
        409,
        trainingService.isRunning()
          ? "Generation is unavailable until training saves a checkpoint and the server loads it."
          : "No trained checkpoint is loaded. Train the model before generating text."
      );
    }

    let text;

    try {
      text = generateText(
        modelService.getModel(),
        modelService.getTokenizer(),
        input.prompt,
        input.maxNewTokens,
        input.temperature
      );
    } catch (error) {
      if (!error.status) {
        error.status = 400;
      }

      throw error;
    }

    const payload = {
      text,
      prompt: input.prompt,
      parameters: {
        maxNewTokens: input.maxNewTokens,
        temperature: input.temperature
      }
    };

    if (trainingService.isRunning()) {
      payload.note = "Generated with the weights already in memory. The training job uses a separate copy and replaces these weights only after it finishes.";
    }

    res.json(payload);
  } catch (error) {
    next(error);
  }
});

module.exports = router;
