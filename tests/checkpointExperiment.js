const GPTModel = require("../models/gptModel");

const {
  loadText,
  createTokenizer
} = require("../tokenizer/tokenizer");

const { prepareData } = require("../data/prepareData");
const { getBatch } = require("../data/nextTokenDataset");
const { trainStep } = require("../training/trainStep");

const {
  saveCheckpoint,
  loadCheckpoint
} = require("../training/checkpoint");

const { softmaxCrossEntropy } = require("../training/loss");

const config = {
  vocabSize: 0,
  dModel: 32,
  numLayers: 1,
  numHeads: 2,
  headDim: 16,
  ffnHidden: 64,
  maxSequenceLength: 64,
  batchSize: 8,
  learningRate: 0.0003
};

// Load Shakespeare and create tokenizer.
const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");
const tokenizer = createTokenizer(text);

config.vocabSize = tokenizer.vocabulary.length;

const data = prepareData(
  tokenizer.encode,
  0.1,
  config.maxSequenceLength
);

// Create the first model.
const modelA = new GPTModel(config);

// Train it for a few steps.
console.log("TRAINING MODEL A");

for (let step = 1; step <= 10; step++) {
  const batch = getBatch(
    data.trainIds,
    config.batchSize,
    config.maxSequenceLength
  );

  const loss = trainStep(
    modelA,
    batch,
    config.batchSize,
    config.maxSequenceLength,
    config.learningRate
  );

  console.log(`Step ${step} | Loss: ${loss.toFixed(4)}`);
}

// Save the trained model.
const checkpointPath =
  "./68-shakespeare-gpt/checkpoints/shakespeare-model.json";

saveCheckpoint(modelA, config, checkpointPath);

// Create a completely fresh model.
console.log("\nCREATING MODEL B");

const modelB = new GPTModel(config);

// Load model A's learned weights into model B.
loadCheckpoint(modelB, checkpointPath);

// Use the same input for both models.
const evaluationBatch = getBatch(
  data.validationIds,
  config.batchSize,
  config.maxSequenceLength
);

const logitsA = modelA.forward(
  evaluationBatch.inputIds,
  config.batchSize,
  config.maxSequenceLength
);

const logitsB = modelB.forward(
  evaluationBatch.inputIds,
  config.batchSize,
  config.maxSequenceLength
);

// Compare the outputs.
let maxDifference = 0;

for (let i = 0; i < logitsA.data.length; i++) {
  const difference = Math.abs(
    logitsA.data[i] - logitsB.data[i]
  );

  if (difference > maxDifference) {
    maxDifference = difference;
  }
}

const lossA = softmaxCrossEntropy(
  logitsA,
  evaluationBatch.targetIds
);

const lossB = softmaxCrossEntropy(
  logitsB,
  evaluationBatch.targetIds
);

console.log("\nCHECKPOINT VERIFICATION");
console.log("-----------------------");
console.log("Model A loss:", lossA.data[0].toFixed(8));
console.log("Model B loss:", lossB.data[0].toFixed(8));
console.log("Maximum output difference:", maxDifference);

if (maxDifference > 0.000000001) {
  throw new Error("Checkpoint verification failed.");
}

console.log("\nPASS: Model weights saved and restored correctly.");