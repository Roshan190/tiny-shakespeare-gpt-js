
const Tensor = require("../tensor/Tensor");
const causalAttention = require("../layers/causalAttention");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// Q, K and V: [batch=1, sequence=3, headDim=2]
const q = new Tensor(
  [0.2, 0.5, 0.7, -0.3, 0.4, 0.8],
  [1, 3, 2]
);

const k = new Tensor(
  [0.6, -0.2, 0.1, 0.9, -0.5, 0.3],
  [1, 3, 2]
);

const v = new Tensor(
  [1, 2, 3, 4, 5, 6],
  [1, 3, 2]
);

// Different coefficients prevent gradients from cancelling.
const coefficients = [
  0.3, -0.7,
  0.5, 0.2,
  -0.4, 0.8
];

// Calculate scalar loss using a fresh forward pass.
function calculateLoss() {
  const output = causalAttention(q, k, v);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// Create differentiable loss for analytical backpropagation.
function createLoss() {
  const output = causalAttention(q, k, v);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  const loss = new Tensor([total], [1]);

  loss.parent = [output];

  loss.backward = () => {
    for (let i = 0; i < output.data.length; i++) {
      output.grad[i] += coefficients[i] * loss.grad[0];
    }
  };

  return loss;
}

// Reset gradients.
q.grad.fill(0);
k.grad.fill(0);
v.grad.fill(0);

// Calculate analytical gradients.
const loss = createLoss();
backward(loss);

// Check every element in Q, K and V.
const tests = [
  { name: "Q", tensor: q },
  { name: "K", tensor: k },
  { name: "V", tensor: v }
];

console.log("======================================");
console.log("CAUSAL ATTENTION GRADIENT CHECK");
console.log("======================================");

console.log("Loss:", loss.data[0].toFixed(8));

let allPassed = true;

for (const test of tests) {
  console.log(`\n${test.name} gradients:`);

  for (let i = 0; i < test.tensor.data.length; i++) {
    const analytical = test.tensor.grad[i];

    const numerical = numericalGradient(
      calculateLoss,
      test.tensor,
      i
    );

    const difference = Math.abs(analytical - numerical);
    const passed = difference < 1e-4;

    if (!passed) {
      allPassed = false;
    }

    console.log(
      `[${i}] ` +
      `analytical=${analytical.toFixed(8)} ` +
      `numerical=${numerical.toFixed(8)} ` +
      `difference=${difference.toFixed(10)} ` +
      `${passed ? "PASS" : "FAIL"}`
    );
  }
}

console.log("\n======================================");
console.log(allPassed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

if (!allPassed) {
  process.exitCode = 1;
}
