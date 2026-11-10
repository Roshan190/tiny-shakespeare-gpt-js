
const Tensor = require("../tensor/Tensor");
const SelfAttentionHead = require("../layers/selfAttentionHead");

const dModel = 4;
const headDim = 2;

const attentionHead = new SelfAttentionHead(
  dModel,
  headDim
);

// Batch=1, sequence=3, features=4
const x = new Tensor(
  [
    0.2, 0.5, 0.1, 0.8,
    0.7, 0.3, 0.9, 0.4,
    0.6, 0.2, 0.5, 0.1
  ],
  [1, 3, dModel]
);

const output = attentionHead.forward(x);

console.log("======================================");
console.log("TRAINABLE SELF-ATTENTION HEAD");
console.log("======================================");

console.log("Input shape:", x.shape);
console.log("Output shape:", output.shape);

console.log(
  "Q shape:",
  attentionHead.query.forward(x).shape
);

console.log(
  "K shape:",
  attentionHead.key.forward(x).shape
);

console.log(
  "V shape:",
  attentionHead.value.forward(x).shape
);

console.log(
  "Total parameters:",
  attentionHead.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  )
);

console.log("Output:", output.data);

if (
  JSON.stringify(output.shape) !==
  JSON.stringify([1, 3, headDim])
) {
  throw new Error("Self-attention shape check failed");
}

console.log("SELF-ATTENTION HEAD CHECK PASSED");
