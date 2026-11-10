const GPTModel = require("../models/gptModel");

const {
  loadText,
  createTokenizer
} = require("../tokenizer/tokenizer");

const { prepareData } = require("../data/prepareData");

const { getBatch } = require("../data/nextTokenDataset");

const { trainStep } = require("../training/trainStep");

const { softmaxCrossEntropy } = require("../training/loss");

// 1. Model configuration.
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

// 2. Load Shakespeare and create tokenizer.
const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");

const tokenizer = createTokenizer(text);

config.vocabSize = tokenizer.vocabulary.length;

// 3. Prepare training data.
const data = prepareData(
  tokenizer.encode,
  0.1,
  config.maxSequenceLength
);

// 4. Create GPT model.
const model = new GPTModel(config);

// 5. Create one fixed batch for evaluation.
const evaluationBatch = getBatch(
  data.trainIds,
  config.batchSize,
  config.maxSequenceLength
);

// Calculate loss without updating weights.
function calculateLoss(batch) {
  const logits = model.forward(
    batch.inputIds,
    config.batchSize,
    config.maxSequenceLength
  );

  const loss = softmaxCrossEntropy(
    logits,
    batch.targetIds
  );

  return loss.data[0];
}

const initialLoss = calculateLoss(evaluationBatch);

console.log("INITIAL FIXED BATCH LOSS:", initialLoss.toFixed(4));

// 6. Train using different random batches.
const totalSteps = 100;
let totalTrainingLoss = 0;

console.log("\nSTARTING RANDOM BATCH TRAINING");

for (let step = 1; step <= totalSteps; step++) {

  // IMPORTANT: A new batch is sampled on every step.
  const batch = getBatch(
    data.trainIds,
    config.batchSize,
    config.maxSequenceLength
  );

  const loss = trainStep(
    model,
    batch,
    config.batchSize,
    config.maxSequenceLength,
    config.learningRate
  );

  if (!Number.isFinite(loss)) {
    throw new Error(`Invalid loss at step ${step}`);
  }

  totalTrainingLoss += loss;

  if (step % 10 === 0) {
    const averageLoss = totalTrainingLoss / 10;

    console.log(
      `Step ${step} | Average batch loss: ${averageLoss.toFixed(4)}`
    );

    totalTrainingLoss = 0;
  }
}

// 7. Evaluate on the same fixed batch.
const finalLoss = calculateLoss(evaluationBatch);

console.log("\nTRAINING SUMMARY");
console.log("----------------");
console.log("Initial fixed batch loss:", initialLoss.toFixed(4));
console.log("Final fixed batch loss:", finalLoss.toFixed(4));

console.log("\nPASS: Random batch training completed.");