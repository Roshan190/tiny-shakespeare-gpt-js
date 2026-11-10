const Tensor = require("../tensor/Tensor");
const TransformerBlock = require("../layers/transformerBlock");

// Temporary placeholders: return the input unchanged.
const identityModule = {
  forward(x) {
    return x;
  },

  parameters() {
    return [];
  }
};

const block = new TransformerBlock(
  4,
  identityModule,
  identityModule
);

const x = new Tensor(
  [1, 2, 3, 4, 5, 6, 7, 8],
  [1, 2, 4]
);

const output = block.forward(x);

console.log("Input shape:", x.shape);
console.log("Output shape:", output.shape);
console.log("Output values:", output.data);

if (
  JSON.stringify(output.shape) !==
  JSON.stringify(x.shape)
) {
  throw new Error("Transformer block shape check failed");
}

console.log("TRANSFORMER BLOCK SHAPE CHECK PASSED");