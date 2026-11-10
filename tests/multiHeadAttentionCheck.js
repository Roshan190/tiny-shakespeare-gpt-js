
const Tensor = require("../tensor/Tensor");
const MultiHeadAttention = require("../layers/multiHeadAttention");

const dModel = 8;
const numHeads = 4;

const attention = new MultiHeadAttention(
  dModel,
  numHeads
);

// Batch=1, sequence=3, features=8
const x = new Tensor(
  [
    0.2, 0.5, 0.1, 0.8, 0.3, 0.6, 0.4, 0.9,
    0.7, 0.3, 0.9, 0.4, 0.5, 0.2, 0.8, 0.1,
    0.6, 0.2, 0.5, 0.1, 0.9, 0.3, 0.7, 0.4
  ],
  [1, 3, dModel]
);

const output = attention.forward(x);

console.log("======================================");
console.log("MULTI-HEAD ATTENTION CHECK");
console.log("======================================");

console.log("Input shape:", x.shape);
console.log("Number of heads:", attention.numHeads);
console.log("Head dimension:", attention.headDim);

console.log("Individual head shapes:");

for (let i = 0; i < attention.heads.length; i++) {
  const headOutput = attention.heads[i].forward(x);

  console.log(`Head ${i + 1}:`, headOutput.shape);
}

console.log("Output shape:", output.shape);

console.log(
  "Total parameters:",
  attention.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  )
);

if (
  JSON.stringify(output.shape) !==
  JSON.stringify(x.shape)
) {
  throw new Error("Multi-head attention shape check failed");
}

console.log("MULTI-HEAD ATTENTION CHECK PASSED");
