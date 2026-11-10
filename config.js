const path = require("path");

const config = {
  // Model
  dModel: 64,
  numLayers: 2,
  numHeads: 4,
  headDim: 16,
  ffnHidden: 256,
  maxSequenceLength: 128,

  // Training
  learningRate: 0.0003,
  batchSize: 16,
  epochs: 10,
  stepsPerEpoch: 100,
  validationSteps: 10,

  // Dataset
  validationRatio: 0.1,

  // Checkpoints and metrics. Paths are anchored to this file so commands
  // work from any working directory.
  checkpointPath: path.join(__dirname, "checkpoints", "shakespeare-model.json"),
  metricsPath: path.join(__dirname, "visualization", "training-metrics.json")
};

module.exports = config;
