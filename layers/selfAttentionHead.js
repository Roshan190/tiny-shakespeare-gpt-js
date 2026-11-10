
const Linear = require("./linear");
const causalAttention = require("./causalAttention");

class SelfAttentionHead {
  constructor(dModel, headDim) {
    this.dModel = dModel;
    this.headDim = headDim;

    // Three independent trainable projections.
    this.query = new Linear(dModel, headDim);
    this.key = new Linear(dModel, headDim);
    this.value = new Linear(dModel, headDim);
  }

  forward(x) {
    // x shape: [B, T, dModel]

    // Generate Q, K and V.
    const q = this.query.forward(x);
    const k = this.key.forward(x);
    const v = this.value.forward(x);

    // Apply causal self-attention.
    const output = causalAttention(q, k, v);

    return output;
  }

  parameters() {
    return [
      ...this.query.parameters(),
      ...this.key.parameters(),
      ...this.value.parameters()
    ];
  }
}

module.exports = SelfAttentionHead;
