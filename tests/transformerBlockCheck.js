
const Tensor = require("../tensor/Tensor");
const TransformerBlock = require("../layers/transformerBlock");
const { backward } = require("../tensor/autograd");

// Small configuration for testing.
const dModel = 8;
const numHeads = 2;
const ffnHidden = 16;

const block = new TransformerBlock(
  dModel,
  numHeads,
  ffnHidden
);

// Batch=1, sequence=3, features=8.
const x = new Tensor(
  [
    0.2, 0.5, 0.1, 0.8, 0.3, 0.6, 0.4, 0.9,
    0.7, 0.3, 0.9, 0.4, 0.5, 0.2, 0.8, 0.1,
    0.6, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4
  ],
  [1, 3, dModel]
);

// Forward pass.
const output = block.forward(x);

console.log("======================================");
console.log("TRANSFORMER BLOCK CHECK");
console.log("======================================");

console.log("Input shape:", x.shape);
console.log("Output shape:", output.shape);

console.log(
  "Number of parameters:",
  block.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  )
);

// Shape validation.
if (
  JSON.stringify(output.shape) !==
  JSON.stringify(x.shape)
) {
  throw new Error("Transformer block shape mismatch");
}

// Reset parameter gradients.
for (const parameter of block.parameters()) {
  parameter.grad.fill(0);
}

// Backpropagate the sum of output values.
// backward(output) seeds every output gradient with 1.
output.grad.fill(0);
backward(output);

// Check that gradients are valid.
const allGradients = [
  ...x.grad,
  ...block.parameters().flatMap(parameter => parameter.grad)
];

if (!allGradients.every(Number.isFinite)) {
  throw new Error("Invalid gradients detected");
}

console.log("Input gradients are finite:", x.grad.every(Number.isFinite));
console.log("Parameter gradients are finite:", block.parameters().every(
  parameter => parameter.grad.every(Number.isFinite)
));

console.log("TRANSFORMER BLOCK CHECK PASSED");
