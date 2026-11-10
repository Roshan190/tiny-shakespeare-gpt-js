const { backward } = require("../tensor/autograd");

const {
  zeroGradients,
  updateParameters
} = require("./optimizer");

const {
  softmaxCrossEntropy
} = require("./loss");

function trainStep(
  model,
  batch,
  batchSize,
  sequenceLength,
  learningRate
) {
  // 1. Get all trainable parameters.
  const parameters = model.parameters();

  // 2. Clear gradients from the previous training step.
  zeroGradients(parameters);

  // 3. Forward pass: predict the next token at every position.
  const logits = model.forward(
    batch.inputIds,
    batchSize,
    sequenceLength
  );

  // 4. Calculate prediction error.
  const loss = softmaxCrossEntropy(
    logits,
    batch.targetIds
  );

  // 5. Backpropagation: calculate gradients.
  backward(loss);

  // 6. Update model parameters using gradient descent.
  updateParameters(parameters, learningRate);

  return loss.data[0];
}

module.exports = { trainStep };