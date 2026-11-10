const fs = require("fs");
const path = require("path");

const projectConfig = require("../../config");
const GPTModel = require("../../models/gptModel");
const { loadText, createTokenizer } = require("../../tokenizer/tokenizer");
const {
  readCheckpoint,
  loadCheckpoint
} = require("../../training/checkpoint");

const corpusPath = path.join(__dirname, "../../data/shakespeare.txt");

const state = {
  tokenizer: null,
  model: null,
  modelConfig: null,
  parameterCount: null,
  checkpointExists: false,
  checkpointValid: false,
  checkpointError: null,
  loadedAt: null
};

function countParameters(model) {
  return model.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  );
}

function parameterCountForConfig(modelConfig) {
  const vocabSize = modelConfig.vocabSize;
  const dModel = modelConfig.dModel;
  const numLayers = modelConfig.numLayers;
  const numHeads = modelConfig.numHeads;
  const headDim = dModel / numHeads;
  const ffnHidden = modelConfig.ffnHidden;
  const maxSequenceLength = modelConfig.maxSequenceLength;

  const tokenEmbedding = vocabSize * dModel;
  const positionEmbedding = maxSequenceLength * dModel;
  const layerNorm = 2 * dModel;
  const attention =
    numHeads * 3 * (dModel * headDim + headDim) +
    (dModel * dModel + dModel);
  const feedForward =
    (dModel * ffnHidden + ffnHidden) +
    (ffnHidden * dModel + dModel);

  return (
    tokenEmbedding +
    positionEmbedding +
    numLayers * (layerNorm + attention + layerNorm + feedForward) +
    layerNorm +
    (dModel * vocabSize + vocabSize)
  );
}

function assertModelConfig(modelConfig) {
  const required = [
    "vocabSize",
    "dModel",
    "numLayers",
    "numHeads",
    "ffnHidden",
    "maxSequenceLength"
  ];

  for (const key of required) {
    if (!Number.isInteger(modelConfig[key]) || modelConfig[key] <= 0) {
      throw new Error(`Checkpoint configuration has an invalid ${key}.`);
    }
  }

  if (modelConfig.dModel % modelConfig.numHeads !== 0) {
    throw new Error("Checkpoint dModel must be divisible by numHeads.");
  }

  const headDim = modelConfig.dModel / modelConfig.numHeads;

  if (
    modelConfig.headDim !== undefined &&
    modelConfig.headDim !== headDim
  ) {
    throw new Error("Checkpoint headDim does not match dModel / numHeads.");
  }
}

function vocabulariesMatch(savedVocabulary, vocabulary) {
  if (!Array.isArray(savedVocabulary)) {
    return true;
  }

  if (savedVocabulary.length !== vocabulary.length) {
    return false;
  }

  for (let i = 0; i < vocabulary.length; i++) {
    if (savedVocabulary[i] !== vocabulary[i]) {
      return false;
    }
  }

  return true;
}

function instantiateFromCheckpoint() {
  const checkpoint = readCheckpoint(projectConfig.checkpointPath);
  const modelConfig = { ...checkpoint.config };

  assertModelConfig(modelConfig);

  if (modelConfig.vocabSize !== state.tokenizer.vocabulary.length) {
    throw new Error(
      `Checkpoint vocabulary size (${modelConfig.vocabSize}) does not match the tokenizer (${state.tokenizer.vocabulary.length}).`
    );
  }

  if (!vocabulariesMatch(modelConfig.vocabulary, state.tokenizer.vocabulary)) {
    throw new Error(
      "Checkpoint vocabulary does not match the current tokenizer."
    );
  }

  const model = new GPTModel(modelConfig);
  loadCheckpoint(model, projectConfig.checkpointPath);

  return {
    model,
    modelConfig,
    parameterCount: countParameters(model)
  };
}

function applyLoadedModel(loaded) {
  state.model = loaded.model;
  state.modelConfig = loaded.modelConfig;
  state.parameterCount = loaded.parameterCount;
  state.checkpointValid = true;
  state.checkpointError = null;
  state.loadedAt = new Date().toISOString();
}

function clearLoadedModel() {
  state.model = null;
  state.modelConfig = null;
  state.parameterCount = null;
  state.checkpointValid = false;
  state.loadedAt = null;
}

function initialize() {
  const text = loadText(corpusPath);
  state.tokenizer = createTokenizer(text);
  state.checkpointExists = fs.existsSync(projectConfig.checkpointPath);

  if (!state.checkpointExists) {
    state.checkpointError = null;
    clearLoadedModel();
    return getStartupSummary();
  }

  try {
    applyLoadedModel(instantiateFromCheckpoint());
  } catch (error) {
    clearLoadedModel();
    state.checkpointError = error.message;
    console.error("Checkpoint could not be loaded:", error.message);
  }

  return getStartupSummary();
}

function reload() {
  state.checkpointExists = fs.existsSync(projectConfig.checkpointPath);

  if (!state.checkpointExists) {
    throw new Error("Checkpoint file is missing.");
  }

  const loaded = instantiateFromCheckpoint();
  applyLoadedModel(loaded);
  return getStartupSummary();
}

function getStartupSummary() {
  return {
    vocabularySize: state.tokenizer.vocabulary.length,
    modelLoaded: state.model !== null,
    checkpointExists: state.checkpointExists,
    checkpointValid: state.checkpointValid,
    checkpointError: state.checkpointError,
    parameterCount: state.parameterCount
  };
}

function isReady() {
  return state.model !== null;
}

function getModel() {
  return state.model;
}

function getTokenizer() {
  return state.tokenizer;
}

function getCheckpointPath() {
  return projectConfig.checkpointPath;
}

function hasValidCheckpoint() {
  return state.checkpointExists && state.checkpointValid && state.model !== null;
}

function getCheckpointError() {
  return state.checkpointError;
}

function getRuntimeState() {
  return {
    modelLoaded: state.model !== null,
    checkpointExists: state.checkpointExists,
    checkpointValid: state.checkpointValid,
    checkpointError: state.checkpointError
  };
}

function describeArchitecture(modelConfig, parameterCount) {
  return {
    vocabSize: modelConfig.vocabSize,
    dModel: modelConfig.dModel,
    numLayers: modelConfig.numLayers,
    numHeads: modelConfig.numHeads,
    headDim: modelConfig.dModel / modelConfig.numHeads,
    ffnHidden: modelConfig.ffnHidden,
    maxSequenceLength: modelConfig.maxSequenceLength,
    parameterCount
  };
}

function getModelInfo() {
  const freshConfig = {
    ...projectConfig,
    vocabSize: state.tokenizer.vocabulary.length
  };

  const activeConfig = state.model ? state.modelConfig : freshConfig;

  return {
    loaded: state.model !== null,
    architecture: "character-level decoder-only GPT",
    source: state.model ? "checkpoint" : "config",
    ...describeArchitecture(
      activeConfig,
      state.model ? state.parameterCount : null
    ),
    freshTraining: {
      ...describeArchitecture(
        freshConfig,
        parameterCountForConfig(freshConfig)
      ),
      batchSize: projectConfig.batchSize,
      learningRate: projectConfig.learningRate,
      epochs: projectConfig.epochs,
      stepsPerEpoch: projectConfig.stepsPerEpoch
    },
    checkpoint: {
      exists: state.checkpointExists,
      valid: state.checkpointValid,
      path: "checkpoints/shakespeare-model.json",
      error: state.checkpointError,
      loadedAt: state.loadedAt
    }
  };
}

module.exports = {
  initialize,
  reload,
  isReady,
  getModel,
  getTokenizer,
  getCheckpointPath,
  hasValidCheckpoint,
  getCheckpointError,
  getRuntimeState,
  getModelInfo,
  getStartupSummary
};
