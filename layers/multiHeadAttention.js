
const Tensor = require("../tensor/Tensor");
const Linear = require("./linear");
const SelfAttentionHead = require("./selfAttentionHead");

// Concatenate multiple attention heads along the feature dimension.
function concatenateHeads(headOutputs) {
  if (headOutputs.length === 0) {
    throw new Error("At least one attention head is required");
  }

  const [batchSize, sequenceLength, headDim] =
    headOutputs[0].shape;

  for (const head of headOutputs) {
    if (
      head.shape.length !== 3 ||
      head.shape[0] !== batchSize ||
      head.shape[1] !== sequenceLength ||
      head.shape[2] !== headDim
    ) {
      throw new Error("All attention heads must have matching shapes");
    }
  }

  const numHeads = headOutputs.length;
  const outputDim = numHeads * headDim;
  const outputData = [];

  // Arrange features head-by-head for each token.
  for (let b = 0; b < batchSize; b++) {
    for (let t = 0; t < sequenceLength; t++) {
      for (let h = 0; h < numHeads; h++) {
        const head = headOutputs[h];

        const start = (b * sequenceLength + t) * headDim;

        for (let d = 0; d < headDim; d++) {
          outputData.push(head.data[start + d]);
        }
      }
    }
  }

  const output = new Tensor(
    outputData,
    [batchSize, sequenceLength, outputDim]
  );

  // Our computation graph uses the singular property: parent.
  output.parent = headOutputs;

  output.backward = () => {
    let outputIndex = 0;

    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < sequenceLength; t++) {
        for (let h = 0; h < numHeads; h++) {
          const head = headOutputs[h];
          const start = (b * sequenceLength + t) * headDim;

          for (let d = 0; d < headDim; d++) {
            head.grad[start + d] += output.grad[outputIndex];
            outputIndex++;
          }
        }
      }
    }
  };

  return output;
}

class MultiHeadAttention {
  constructor(dModel, numHeads) {
    if (dModel % numHeads !== 0) {
      throw new Error("dModel must be divisible by numHeads");
    }

    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;

    // Each head has independent Q, K and V projections.
    this.heads = [];

    for (let i = 0; i < numHeads; i++) {
      this.heads.push(
        new SelfAttentionHead(dModel, this.headDim)
      );
    }

    // Final projection after concatenating all heads.
    this.outputProjection = new Linear(dModel, dModel);
  }

  forward(x) {
    // Run every head on the same input.
    const headOutputs = [];

    for (const head of this.heads) {
      headOutputs.push(head.forward(x));
    }

    // [B, T, numHeads * headDim]
    const combined = concatenateHeads(headOutputs);

    // [B, T, dModel]
    return this.outputProjection.forward(combined);
  }

  parameters() {
    const result = [];

    for (const head of this.heads) {
      result.push(...head.parameters());
    }

    result.push(...this.outputProjection.parameters());

    return result;
  }
}

module.exports = MultiHeadAttention;
