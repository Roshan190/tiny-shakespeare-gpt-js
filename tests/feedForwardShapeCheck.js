
const Tensor = require("../tensor/Tensor");
const FeedForward = require("../layers/feedForward");

const dModel = 128;
const hiddenSize = 512;

const feedForward = new FeedForward(
  dModel,
  hiddenSize
);

const x = new Tensor(
  new Array(2 * 10 * dModel).fill(0.5),
  [2, 10, dModel]
);

const output = feedForward.forward(x);

console.log("======================================");
console.log("FEED-FORWARD NETWORK SHAPE CHECK");
console.log("======================================");

console.log("Input shape:", x.shape);
console.log("Output shape:", output.shape);

console.log(
  "Number of parameters:",
  feedForward.parameters().reduce(
    (total, parameter) => total + parameter.data.length,
    0
  )
);

console.log("First 10 output values:", output.data.slice(0, 10));

if (
  JSON.stringify(output.shape) !==
  JSON.stringify(x.shape)
) {
  throw new Error("FeedForward shape check failed");
}

console.log("FEED-FORWARD SHAPE CHECK PASSED");
