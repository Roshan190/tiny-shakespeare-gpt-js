
const Tensor = require("../tensor/Tensor");
const { softmaxCrossEntropy } = require("../training/loss");
const { backward } = require("../tensor/autograd");

// Batch=1, sequence=2, vocabulary=3.
const logits = new Tensor(
  [
    1.2, -0.5, 2.1,
    0.3, 1.5, -0.2
  ],
  [1, 2, 3]
);

// Correct next-token IDs.
const targets = [2, 1];

const loss = softmaxCrossEntropy(logits, targets);

console.log("======================================");
console.log("SOFTMAX CROSS-ENTROPY CHECK");
console.log("======================================");

console.log("Logits shape:", logits.shape);
console.log("Targets:", targets);
console.log("Loss:", loss.data[0].toFixed(8));

// Backpropagation.
logits.grad.fill(0);
backward(loss);

console.log("Logit gradients:", logits.grad);

// Each position's gradients should sum approximately to zero.
for (let token = 0; token < 2; token++) {
  const start = token * 3;

  const gradientSum =
    logits.grad[start] +
    logits.grad[start + 1] +
    logits.grad[start + 2];

  console.log(
    `Token ${token} gradient sum:`,
    gradientSum.toFixed(10)
  );

  if (Math.abs(gradientSum) > 1e-8) {
    throw new Error("Softmax gradient sum should be zero");
  }
}

if (!Number.isFinite(loss.data[0])) {
  throw new Error("Loss is not finite");
}

console.log("SOFTMAX CROSS-ENTROPY CHECK PASSED");
