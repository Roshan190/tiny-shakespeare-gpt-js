
const Tensor = require("../tensor/Tensor");
const FeedForward = require("../layers/feedForward");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// Small network for faster gradient checking.
const dModel = 2;
const hiddenSize = 3;

// Input: batch=1, sequence=1, features=2
const x = new Tensor(
  [0.5, -0.8],
  [1, 1, 2]
);

const feedForward = new FeedForward(dModel, hiddenSize);

// Weights used to calculate a scalar loss.
const coefficients = [0.3, -0.7];

// Forward pass used by numerical gradient checking.
function calculateLoss() {
  const output = feedForward.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// Build a differentiable scalar loss for backpropagation.
function createLoss() {
  const output = feedForward.forward(x);

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

// 1. Reset gradients.
x.grad.fill(0);

const parameters = feedForward.parameters();

for (const parameter of parameters) {
  parameter.grad.fill(0);
}

// 2. Calculate analytical gradients.
const loss = createLoss();

backward(loss);

// 3. Prepare gradient checks.
const tests = [
  {
    name: "Input x[0]",
    tensor: x,
    index: 0
  },
  {
    name: "Linear1 weight[0]",
    tensor: feedForward.linear1.weight,
    index: 0
  },
  {
    name: "Linear1 bias[0]",
    tensor: feedForward.linear1.bias,
    index: 0
  },
  {
    name: "Linear2 weight[1]",
    tensor: feedForward.linear2.weight,
    index: 1
  },
  {
    name: "Linear2 bias[0]",
    tensor: feedForward.linear2.bias,
    index: 0
  }
];

// 4. Compare analytical and numerical gradients.
console.log("======================================");
console.log("FEED-FORWARD GRADIENT CHECK");
console.log("======================================");

console.log("Loss:", loss.data[0].toFixed(8));

let allPassed = true;

for (const test of tests) {
  const analytical = test.tensor.grad[test.index];

  const numerical = numericalGradient(
    calculateLoss,
    test.tensor,
    test.index
  );

  const difference = Math.abs(analytical - numerical);
  const passed = difference < 1e-4;

  if (!passed) {
    allPassed = false;
  }

  console.log(
    `${test.name}: ` +
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
