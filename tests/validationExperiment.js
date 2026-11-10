const GPTModel = require("../models/gptModel");

const {
  loadText,
  createTokenizer
} = require("../tokenizer/tokenizer");

const { prepareData } = require("../data/prepareData");

const { getBatch } = require("../data/nextTokenDataset");

const { trainStep } = require("../training/trainStep");

const { softmaxCrossEntropy } = require("../training/loss");

// 1. Configuration.
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

// 2. Load and tokenize Shakespeare.
const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");

const tokenizer = createTokenizer(text);

config.vocabSize = tokenizer.vocabulary.length;

// 3. Split training and validation data.
const data = prepareData(
  tokenizer.encode,
  0.1,
  config.maxSequenceLength
);

// 4. Initialize GPT.
const model = new GPTModel(config);

// 5. Create fixed evaluation batches.
const trainingBatch = getBatch(
  data.trainIds,
  config.batchSize,
  config.maxSequenceLength
);

const validationBatch = getBatch(
  data.validationIds,
  config.batchSize,
  config.maxSequenceLength
);

// 6. Calculate loss without updating weights.
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

// Initial measurements.
const initialTrainingLoss = calculateLoss(trainingBatch);
const initialValidationLoss = calculateLoss(validationBatch);

console.log("INITIAL EVALUATION");
console.log("------------------");
console.log("Training loss:", initialTrainingLoss.toFixed(4));
console.log("Validation loss:", initialValidationLoss.toFixed(4));

// 7. Train only on training data.
console.log("\nSTARTING TRAINING");

const totalSteps = 100;

for (let step = 1; step <= totalSteps; step++) {
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
    throw new Error(`Invalid training loss at step ${step}`);
  }

  if (step % 10 === 0) {
    console.log(
      `Training step ${step} | Loss: ${loss.toFixed(4)}`
    );
  }
}

// 8. Evaluate again using the same fixed batches.
const finalTrainingLoss = calculateLoss(trainingBatch);
const finalValidationLoss = calculateLoss(validationBatch);

console.log("\nFINAL EVALUATION");
console.log("----------------");
console.log("Training loss:", finalTrainingLoss.toFixed(4));
console.log("Validation loss:", finalValidationLoss.toFixed(4));

console.log("\nLOSS COMPARISON");
console.log("---------------");

console.log(
  "Training loss change:",
  (initialTrainingLoss - finalTrainingLoss).toFixed(4)
);

console.log(
  "Validation loss change:",
  (initialValidationLoss - finalValidationLoss).toFixed(4)
);