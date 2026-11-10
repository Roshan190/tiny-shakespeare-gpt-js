
const Linear = require("./linear");
const gelu = require("./gelu");

class FeedForward {
  constructor(dModel, hiddenSize) {
    // First projection: dModel -> hiddenSize
    this.linear1 = new Linear(dModel, hiddenSize);

    // Second projection: hiddenSize -> dModel
    this.linear2 = new Linear(hiddenSize, dModel);
  }

  forward(x) {
    // Step 1: Expand the feature dimensions.
    const hidden = this.linear1.forward(x);

    // Step 2: Apply non-linearity.
    const activated = gelu(hidden);

    // Step 3: Project back to the original dimensions.
    const output = this.linear2.forward(activated);

    return output;
  }

  parameters() {
    return [
      ...this.linear1.parameters(),
      ...this.linear2.parameters()
    ];
  }
}

module.exports = FeedForward;
