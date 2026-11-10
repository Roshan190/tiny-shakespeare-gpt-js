const GPTModel = require("../models/gptModel");

const {
  loadText,
  createTokenizer
} = require("../tokenizer/tokenizer");

const { prepareData } = require("../data/prepareData");

const { getBatch } = require("../data/nextTokenDataset");

const { trainStep } = require("../training/trainStep");

const { softmaxCrossEntropy } = require("../training/loss");

// Experiment configuration.
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

// Load dataset.
const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");

const tokenizer = createTokenizer(text);

config.vocabSize = tokenizer.vocabulary.length;

const data = prepareData(
  tokenizer.encode,
  0.1,
  config.maxSequenceLength
);

// Create model.
const model = new GPTModel(config);

// Use one fixed batch to compare loss consistently.
const batch = getBatch(
  data.trainIds,
  config.batchSize,
  config.maxSequenceLength
);

function calculateLoss() {
  const logits = model.forward(
    batch.inputIds,
    config.batchSize,
    config.maxSequenceLength
  );

  return softmaxCrossEntropy(
    logits,
    batch.targetIds
  ).data[0];
}

console.log("INITIAL LOSS:", calculateLoss());

console.log("\nSTARTING TRAINING");

for (let step = 1; step <= 100; step++) {
  const loss = trainStep(
    model,
    batch,
    config.batchSize,
    config.maxSequenceLength,
    config.learningRate
  );

  if (step % 10 === 0) {
    console.log(
      `Step ${step} | Training loss: ${loss.toFixed(4)}`
    );
  }
}

console.log("\nFINAL LOSS:", calculateLoss());