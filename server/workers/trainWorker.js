const path = require("path");
const { parentPort, workerData } = require("worker_threads");

const projectConfig = require("../../config");
const GPTModel = require("../../models/gptModel");
const { loadText, createTokenizer } = require("../../tokenizer/tokenizer");
const { prepareData } = require("../../data/prepareData");
const { trainModel } = require("../../training/trainLoop");
const {
  readCheckpoint,
  loadCheckpoint,
  saveCheckpoint
} = require("../../training/checkpoint");
const { saveMetrics } = require("../../visualization/saveMetrics");

function positiveInteger(value, fallback) {
  return Number.isInteger(value) && value > 0 ? value : fallback;
}

function buildSavedConfig(modelConfig, tokenizer) {
  return {
    vocabSize: modelConfig.vocabSize,
    dModel: modelConfig.dModel,
    numLayers: modelConfig.numLayers,
    numHeads: modelConfig.numHeads,
    headDim: modelConfig.dModel / modelConfig.numHeads,
    ffnHidden: modelConfig.ffnHidden,
    maxSequenceLength: modelConfig.maxSequenceLength,
    batchSize: modelConfig.batchSize,
    learningRate: modelConfig.learningRate,
    epochs: modelConfig.epochs,
    stepsPerEpoch: modelConfig.stepsPerEpoch,
    validationSteps: modelConfig.validationSteps,
    validationRatio: modelConfig.validationRatio,
    vocabulary: tokenizer.vocabulary
  };
}

function createModelConfig(tokenizer) {
  const { mode, epochs, stepsPerEpoch, learningRate, checkpointPath } = workerData;

  if (mode === "resume") {
    const saved = readCheckpoint(checkpointPath);

    if (saved.config.vocabSize !== tokenizer.vocabulary.length) {
      throw new Error(
        "Checkpoint vocabulary does not match the current tokenizer."
      );
    }

    if (Array.isArray(saved.config.vocabulary)) {
      const sameLength = saved.config.vocabulary.length === tokenizer.vocabulary.length;
      const sameCharacters = sameLength && saved.config.vocabulary.every(
        (character, index) => character === tokenizer.vocabulary[index]
      );

      if (!sameCharacters) {
        throw new Error(
          "Checkpoint vocabulary does not match the current tokenizer."
        );
      }
    }

    return {
      vocabSize: saved.config.vocabSize,
      dModel: saved.config.dModel,
      numLayers: saved.config.numLayers,
      numHeads: saved.config.numHeads,
      headDim: saved.config.dModel / saved.config.numHeads,
      ffnHidden: saved.config.ffnHidden,
      maxSequenceLength: saved.config.maxSequenceLength,
      batchSize: positiveInteger(saved.config.batchSize, projectConfig.batchSize),
      learningRate,
      epochs,
      stepsPerEpoch,
      validationSteps: positiveInteger(
        saved.config.validationSteps,
        projectConfig.validationSteps
      ),
      validationRatio: Number.isFinite(saved.config.validationRatio)
        ? saved.config.validationRatio
        : projectConfig.validationRatio
    };
  }

  return {
    vocabSize: tokenizer.vocabulary.length,
    dModel: projectConfig.dModel,
    numLayers: projectConfig.numLayers,
    numHeads: projectConfig.numHeads,
    headDim: projectConfig.dModel / projectConfig.numHeads,
    ffnHidden: projectConfig.ffnHidden,
    maxSequenceLength: projectConfig.maxSequenceLength,
    batchSize: projectConfig.batchSize,
    learningRate,
    epochs,
    stepsPerEpoch,
    validationSteps: projectConfig.validationSteps,
    validationRatio: projectConfig.validationRatio
  };
}

function main() {
  const text = loadText(
    path.join(__dirname, "../../data/shakespeare.txt")
  );
  const tokenizer = createTokenizer(text);
  const modelConfig = createModelConfig(tokenizer);
  const model = new GPTModel(modelConfig);

  if (workerData.mode === "resume") {
    loadCheckpoint(model, workerData.checkpointPath);
  }

  const data = prepareData(
    tokenizer.encode,
    modelConfig.validationRatio,
    modelConfig.maxSequenceLength
  );

  const savedConfig = buildSavedConfig(modelConfig, tokenizer);

  const history = trainModel(model, data, modelConfig, (progress) => {
    if (progress.phase === "epoch") {
      saveCheckpoint(model, savedConfig, workerData.checkpointPath);
    }

    parentPort.postMessage({ type: "progress", progress });
  });

  saveCheckpoint(
    model,
    savedConfig,
    workerData.checkpointPath
  );

  saveMetrics(history);

  parentPort.postMessage({ type: "complete", history });
}

try {
  main();
} catch (error) {
  parentPort.postMessage({
    type: "error",
    message: error && error.message ? error.message : String(error)
  });
  process.exitCode = 1;
}
