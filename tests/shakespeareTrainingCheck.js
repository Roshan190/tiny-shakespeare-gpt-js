const config = require("../config");

const GPTModel = require("../models/gptModel");

const {
  loadText,
  createTokenizer
} = require("../tokenizer/tokenizer");

const { prepareData } = require("../data/prepareData");

const { getBatch } = require("../data/nextTokenDataset");

const { trainStep } = require("../training/trainStep");

// 1. Load Shakespeare text.
const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");

// 2. Create tokenizer.
const tokenizer = createTokenizer(text);

// 3. Update vocabulary size dynamically.
const modelConfig = {
  ...config,
  vocabSize: tokenizer.vocabulary.length
};

// 4. Prepare dataset.
const data = prepareData(
  tokenizer.encode,
  0.1,
  modelConfig.maxSequenceLength
);

// 5. Create GPT model.
const model = new GPTModel(modelConfig);

// 6. Generate one real Shakespeare batch.
const batch = getBatch(
  data.trainIds,
  modelConfig.batchSize,
  modelConfig.maxSequenceLength
);

console.log("SHAKESPEARE TRAINING CHECK");
console.log("--------------------------");

console.log("Vocabulary size:", modelConfig.vocabSize);
console.log("Batch size:", modelConfig.batchSize);
console.log("Sequence length:", modelConfig.maxSequenceLength);

console.log("\nFirst 20 input IDs:");
console.log(batch.inputIds.slice(0, 20));

console.log("\nFirst 20 target IDs:");
console.log(batch.targetIds.slice(0, 20));

console.log("\nDecoded input:");
console.log(tokenizer.decode(batch.inputIds.slice(0, 100)));

console.log("\nDecoded targets:");
console.log(tokenizer.decode(batch.targetIds.slice(0, 100)));

// 7. Perform one training step.
console.log("\nRunning one training step...");

const loss = trainStep(
  model,
  batch,
  modelConfig.batchSize,
  modelConfig.maxSequenceLength,
  modelConfig.learningRate
);

console.log("\nTRAINING RESULT");
console.log("---------------");
console.log("Loss:", loss);

if (!Number.isFinite(loss)) {
  throw new Error("Training loss is invalid.");
}

console.log("\nPASS: First Shakespeare training step completed.");