
const Tensor = require("../tensor/Tensor");
const {
  softmaxCrossEntropy
} = require("../training/loss");

const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

const logits = new Tensor(
  [
    1.2, -0.5, 2.1,
    0.3, 1.5, -0.2
  ],
  [1, 2, 3]
);

const targets = [2, 1];

function calculateLoss() {
  return softmaxCrossEntropy(logits, targets).data[0];
}

// Analytical gradients.
logits.grad.fill(0);

const loss = softmaxCrossEntropy(logits, targets);

backward(loss);

console.log("======================================");
console.log("CROSS-ENTROPY GRADIENT CHECK");
console.log("======================================");

let allPassed = true;

for (let i = 0; i < logits.data.length; i++) {
  const analytical = logits.grad[i];

  const numerical = numericalGradient(
    calculateLoss,
    logits,
    i
  );

  const difference = Math.abs(analytical - numerical);
  const passed = difference < 1e-5;

  if (!passed) allPassed = false;

  console.log(
    `Logit[${i}]: ` +
    `analytical=${analytical.toFixed(8)} ` +
    `numerical=${numerical.toFixed(8)} ` +
    `difference=${difference.toFixed(10)} ` +
    `${passed ? "PASS" : "FAIL"}`
  );
}

console.log("======================================");
console.log(allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

if (!allPassed) {
  process.exitCode = 1;
}
