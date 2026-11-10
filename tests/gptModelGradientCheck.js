
const Tensor = require("../tensor/Tensor");
const GPTModel = require("../models/gptModel");
const config = require("../config");
const { backward } = require("../tensor/autograd");

// Small model for testing.
const testConfig = {
  ...config,
  vocabSize: 12,
  dModel: 8,
  numLayers: 2,
  numHeads: 2,
  headDim: 4,
  ffnHidden: 16,
  maxSequenceLength: 8
};

const model = new GPTModel(testConfig);

// Input: batch=1, sequence=3.
const tokenIds = [1, 4, 7];

const batchSize = 1;
const sequenceLength = 3;

// Reset parameter gradients.
const parameters = model.parameters();

for (const parameter of parameters) {
  parameter.grad.fill(0);
}

// Forward pass.
const logits = model.forward(
  tokenIds,
  batchSize,
  sequenceLength
);

// Build a simple scalar objective from the logits.
let total = 0;

for (let i = 0; i < logits.data.length; i++) {
  const coefficient = ((i % 7) - 3) * 0.1;

  total += logits.data[i] * coefficient;
}

const loss = new Tensor([total], [1]);

loss.parent = [logits];

loss.backward = () => {
  for (let i = 0; i < logits.data.length; i++) {
    const coefficient = ((i % 7) - 3) * 0.1;

    logits.grad[i] += coefficient * loss.grad[0];
  }
};

// Backpropagate through the entire GPT model.
backward(loss);

// Check gradients.
let finiteParameterCount = 0;
let nonZeroParameterCount = 0;

for (const parameter of parameters) {
  for (const gradient of parameter.grad) {
    if (!Number.isFinite(gradient)) {
      throw new Error("Invalid parameter gradient detected");
    }

    finiteParameterCount++;

    if (Math.abs(gradient) > 1e-12) {
      nonZeroParameterCount++;
    }
  }
}

const embeddingGradients = model.tokenEmbedding.weight.grad;

const embeddingHasGradient = embeddingGradients.some(
  gradient => Math.abs(gradient) > 1e-12
);

console.log("======================================");
console.log("FULL GPT BACKPROPAGATION CHECK");
console.log("======================================");

console.log("Logits shape:", logits.shape);
console.log("Loss:", loss.data[0].toFixed(8));

console.log("Total parameters:", parameters.reduce(
  (total, parameter) => total + parameter.data.length,
  0
));

console.log("Finite parameter gradients:", finiteParameterCount);
console.log("Non-zero parameter gradients:", nonZeroParameterCount);

console.log(
  "Token embedding receives gradients:",
  embeddingHasGradient
);

if (!embeddingHasGradient) {
  throw new Error("Token embedding did not receive gradients");
}

if (finiteParameterCount === 0) {
  throw new Error("No parameter gradients were calculated");
}

console.log("FULL GPT BACKPROPAGATION CHECK PASSED");
