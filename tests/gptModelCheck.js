
const GPTModel = require("../models/gptModel");
const config = require("../config");

// Use a small test configuration.
const testConfig = {
  ...config,
  vocabSize: 20,
  dModel: 8,
  numLayers: 2,
  numHeads: 2,
  headDim: 4,
  ffnHidden: 16,
  maxSequenceLength: 16
};

const model = new GPTModel(testConfig);

// Batch=2, sequence=3.
// Each number represents a token ID.
const tokenIds = [
  1, 4, 7,
  2, 5, 9
];

const logits = model.forward(tokenIds, 2, 3);

console.log("======================================");
console.log("GPT MODEL CHECK");
console.log("======================================");

console.log("Vocabulary size:", model.vocabSize);
console.log("Transformer blocks:", model.blocks.length);

console.log("Input token count:", tokenIds.length);
console.log("Logits shape:", logits.shape);

console.log(
  "Total parameters:",
  model.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  )
);

console.log("First 10 logits:", logits.data.slice(0, 10));

// Expected: [batch=2, sequence=3, vocabSize=20].
const expectedShape = [2, 3, testConfig.vocabSize];

if (
  JSON.stringify(logits.shape) !==
  JSON.stringify(expectedShape)
) {
  throw new Error("GPT model shape check failed");
}

if (!logits.data.every(Number.isFinite)) {
  throw new Error("GPT model produced invalid logits");
}

console.log("GPT MODEL CHECK PASSED");
