
const LayerNorm = require("./layerNorm");
const MultiHeadAttention = require("./multiHeadAttention");
const FeedForward = require("./feedForward");
const { add } = require("../tensor/operation");

class TransformerBlock {
  constructor(dModel, numHeads, ffnHidden) {
    this.dModel = dModel;

    // First normalization.
    this.norm1 = new LayerNorm(dModel);

    // Real multi-head causal attention.
    this.attention = new MultiHeadAttention(
      dModel,
      numHeads
    );

    // Second normalization.
    this.norm2 = new LayerNorm(dModel);

    // Feed-forward network.
    this.feedForward = new FeedForward(
      dModel,
      ffnHidden
    );
  }

  forward(x) {
    // 1. Normalize input before attention.
    const normalized1 = this.norm1.forward(x);

    // 2. Apply multi-head causal attention.
    const attentionOutput = this.attention.forward(normalized1);

    // 3. First residual connection.
    const h = add(x, attentionOutput);

    // 4. Normalize before feed-forward.
    const normalized2 = this.norm2.forward(h);

    // 5. Apply feed-forward network.
    const feedForwardOutput = this.feedForward.forward(normalized2);

    // 6. Second residual connection.
    const output = add(h, feedForwardOutput);

    return output;
  }

  parameters() {
    return [
      ...this.norm1.parameters(),
      ...this.attention.parameters(),
      ...this.norm2.parameters(),
      ...this.feedForward.parameters()
    ];
  }
}

module.exports = TransformerBlock;
