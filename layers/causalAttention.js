
const Tensor = require("../tensor/Tensor");

function causalAttention(q, k, v) {
  const [batchSize, sequenceLength, headDim] = q.shape;

  if (
    q.shape.length !== 3 ||
    k.shape.length !== 3 ||
    v.shape.length !== 3 ||
    JSON.stringify(q.shape) !== JSON.stringify(k.shape) ||
    JSON.stringify(q.shape) !== JSON.stringify(v.shape)
  ) {
    throw new Error("Q, K and V must have matching [B, T, D] shapes");
  }

  const scale = 1 / Math.sqrt(headDim);

  // Attention weights: [B, T, T]
  const weights = new Array(
    batchSize * sequenceLength * sequenceLength
  ).fill(0);

  // Output: [B, T, D]
  const outputData = new Array(
    batchSize * sequenceLength * headDim
  ).fill(0);

  // --------------------------------
  // 1. Calculate attention scores
  // 2. Apply causal mask
  // 3. Apply softmax
  // --------------------------------

  for (let b = 0; b < batchSize; b++) {
    for (let i = 0; i < sequenceLength; i++) {
      const scores = [];

      // Calculate scores only for allowed positions.
      for (let j = 0; j <= i; j++) {
        let score = 0;

        for (let d = 0; d < headDim; d++) {
          const qIndex =
            (b * sequenceLength + i) * headDim + d;

          const kIndex =
            (b * sequenceLength + j) * headDim + d;

          score += q.data[qIndex] * k.data[kIndex];
        }

        scores.push(score * scale);
      }

      // Numerically stable softmax.
      const maxScore = Math.max(...scores);

      let denominator = 0;

      for (let j = 0; j < scores.length; j++) {
        scores[j] = Math.exp(scores[j] - maxScore);
        denominator += scores[j];
      }

      // Store attention weights.
      for (let j = 0; j <= i; j++) {
        const weightIndex =
          (b * sequenceLength + i) * sequenceLength + j;

        weights[weightIndex] = scores[j] / denominator;
      }
    }
  }

  // --------------------------------
  // 4. Multiply attention weights by V
  // --------------------------------

  for (let b = 0; b < batchSize; b++) {
    for (let i = 0; i < sequenceLength; i++) {
      for (let d = 0; d < headDim; d++) {
        let total = 0;

        for (let j = 0; j <= i; j++) {
          const weightIndex =
            (b * sequenceLength + i) * sequenceLength + j;

          const vIndex =
            (b * sequenceLength + j) * headDim + d;

          total += weights[weightIndex] * v.data[vIndex];
        }

        const outputIndex =
          (b * sequenceLength + i) * headDim + d;

        outputData[outputIndex] = total;
      }
    }
  }

  const output = new Tensor(
    outputData,
    [batchSize, sequenceLength, headDim]
  );

  // --------------------------------
  // 5. Backpropagation
  // --------------------------------

  output.parent = [q, k, v];

  output.backward = () => {
    for (let b = 0; b < batchSize; b++) {
      for (let i = 0; i < sequenceLength; i++) {
        // Gradient with respect to attention weights.
        const weightGradients = new Array(i + 1).fill(0);

        for (let j = 0; j <= i; j++) {
          for (let d = 0; d < headDim; d++) {
            const outputIndex =
              (b * sequenceLength + i) * headDim + d;

            const vIndex =
              (b * sequenceLength + j) * headDim + d;

            weightGradients[j] +=
              output.grad[outputIndex] * v.data[vIndex];
          }
        }

        // Softmax backward.
        let weightedGradientSum = 0;

        for (let j = 0; j <= i; j++) {
          const weightIndex =
            (b * sequenceLength + i) * sequenceLength + j;

          weightedGradientSum +=
            weights[weightIndex] * weightGradients[j];
        }

        // Backpropagate through scores into Q and K.
        for (let j = 0; j <= i; j++) {
          const weightIndex =
            (b * sequenceLength + i) * sequenceLength + j;

          const weight = weights[weightIndex];

          const scoreGradient =
            weight * (weightGradients[j] - weightedGradientSum);

          for (let d = 0; d < headDim; d++) {
            const qIndex =
              (b * sequenceLength + i) * headDim + d;

            const kIndex =
              (b * sequenceLength + j) * headDim + d;

            q.grad[qIndex] += scoreGradient * k.data[kIndex] * scale;
            k.grad[kIndex] += scoreGradient * q.data[qIndex] * scale;
          }
        }

        // Backpropagate through weighted sum into V.
        for (let j = 0; j <= i; j++) {
          const weightIndex =
            (b * sequenceLength + i) * sequenceLength + j;

          const weight = weights[weightIndex];

          for (let d = 0; d < headDim; d++) {
            const outputIndex =
              (b * sequenceLength + i) * headDim + d;

            const vIndex =
              (b * sequenceLength + j) * headDim + d;

            v.grad[vIndex] +=
              weight * output.grad[outputIndex];
          }
        }
      }
    }
  };

  return output;
}

module.exports = causalAttention;
