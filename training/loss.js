
const Tensor = require("../tensor/Tensor");

function meanSquaredError(predictions, targets) {
  if (predictions.data.length !== targets.length) {
    throw new Error("Prediction and target sizes must match");
  }

  const count = targets.length;
  let total = 0;

  for (let i = 0; i < count; i++) {
    const difference = predictions.data[i] - targets[i];
    total += difference * difference;
  }

  const loss = new Tensor([total / count], [1]);

  loss.parent = [predictions];

  loss.backward = () => {
    for (let i = 0; i < count; i++) {
      const difference = predictions.data[i] - targets[i];

      predictions.grad[i] +=
        loss.grad[0] * 2 * difference / count;
    }
  };

  return loss;
}

function softmaxCrossEntropy(logits, targets) {
  if (logits.shape.length !== 3) {
    throw new Error("Logits must have shape [batch, sequence, vocabSize]");
  }

  const [batchSize, sequenceLength, vocabSize] = logits.shape;
  const tokenCount = batchSize * sequenceLength;

  if (targets.length !== tokenCount) {
    throw new Error("Target count must equal batchSize * sequenceLength");
  }

  for (const target of targets) {
    if (
      !Number.isInteger(target) ||
      target < 0 ||
      target >= vocabSize
    ) {
      throw new Error(`Invalid target token ID: ${target}`);
    }
  }

  const probabilities = new Array(logits.data.length).fill(0);

  let totalLoss = 0;

  // Calculate softmax and cross-entropy for every token position.
  for (let token = 0; token < tokenCount; token++) {
    const start = token * vocabSize;

    // Find the maximum logit for numerical stability.
    let maxLogit = -Infinity;

    for (let v = 0; v < vocabSize; v++) {
      maxLogit = Math.max(maxLogit, logits.data[start + v]);
    }

    // Calculate exponentials and their sum.
    let denominator = 0;

    for (let v = 0; v < vocabSize; v++) {
      const value = Math.exp(logits.data[start + v] - maxLogit);

      probabilities[start + v] = value;
      denominator += value;
    }

    // Convert exponentials into probabilities.
    for (let v = 0; v < vocabSize; v++) {
      probabilities[start + v] /= denominator;
    }

    // Cross-entropy for the correct target.
    const target = targets[token];

    const targetLogit = logits.data[start + target];

    totalLoss +=
      Math.log(denominator) + maxLogit - targetLogit;
  }

  // Average loss across all token positions.
  const averageLoss = totalLoss / tokenCount;

  const loss = new Tensor([averageLoss], [1]);

  // Connect loss to logits in the computation graph.
  loss.parent = [logits];

  // Backpropagation:
  // dL/dLogits = (probability - oneHotTarget) / tokenCount
  loss.backward = () => {
    for (let token = 0; token < tokenCount; token++) {
      const start = token * vocabSize;
      const target = targets[token];

      for (let v = 0; v < vocabSize; v++) {
        const targetValue = v === target ? 1 : 0;

        logits.grad[start + v] +=
          loss.grad[0] *
          (probabilities[start + v] - targetValue) /
          tokenCount;
      }
    }
  };

  return loss;
}

module.exports = {
  meanSquaredError,
  softmaxCrossEntropy
};
