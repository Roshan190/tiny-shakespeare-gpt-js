const GPTModel = require("../models/gptModel");

const { getBatch } = require("../data/nextTokenDataset");
const { trainStep } = require("../training/trainStep");

const config = {
  vocabSize: 12,
  dModel: 8,
  numLayers: 1,
  numHeads: 2,
  headDim: 4,
  ffnHidden: 16,
  maxSequenceLength: 8
};

// A small artificial token sequence.
const tokenIds = [
  1, 2, 3, 4, 5, 6, 7, 8,
  2, 3, 4, 5, 6, 7, 8, 9,
  3, 4, 5, 6, 7, 8, 9, 10,
  4, 5, 6, 7, 8, 9, 10, 11
];

const batchSize = 2;
const sequenceLength = 4;
const learningRate = 0.0001;

// Create the model.
const model = new GPTModel(config);

// Generate a training batch.
const batch = getBatch(
  tokenIds,
  batchSize,
  sequenceLength
);

console.log("INPUT IDS:", batch.inputIds);
console.log("TARGET IDS:", batch.targetIds);

// Run one training step.
const loss = trainStep(
  model,
  batch,
  batchSize,
  sequenceLength,
  learningRate
);

console.log("\nTRAINING STEP RESULT");
console.log("--------------------");
console.log("Batch size:", batchSize);
console.log("Sequence length:", sequenceLength);
console.log("Vocabulary size:", config.vocabSize);
console.log("Loss:", loss);

// Basic validation.
if (!Number.isFinite(loss)) {
  throw new Error("Training loss is not finite!");
}

console.log("\nPASS: One complete training step executed.");