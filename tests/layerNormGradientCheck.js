
const Tensor = require("../tensor/Tensor");
const LayerNorm = require("../layers/layerNorm");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// 1. Create the input: batch=1, sequence=2, features=4
const x = new Tensor(
  [2, 4, 6, 8, 1, 3, 2, 7],
  [1, 2, 4]
);

// 2. Create LayerNorm
const layerNorm = new LayerNorm(4);

// Different weights make the loss sensitive to different features.
const coefficients = [
  0.2, -0.4, 0.7, 0.1,
  -0.3, 0.8, 0.5, -0.2
];

// 3. Calculate a scalar loss from the LayerNorm output.
function calculateLoss() {
  const output = layerNorm.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// 4. Build a differentiable scalar loss.
function createLossTensor() {
  const output = layerNorm.forward(x);
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

// 5. Calculate analytical gradients using backpropagation.
x.grad.fill(0);
layerNorm.gamma.grad.fill(0);
layerNorm.beta.grad.fill(0);

const loss = createLossTensor();
backward(loss);

// 6. Compare analytical gradients against numerical gradients.
const tests = [
  {
    name: "Input x[0]",
    tensor: x,
    index: 0
  },
  {
    name: "Gamma[1]",
    tensor: layerNorm.gamma,
    index: 1
  },
  {
    name: "Beta[2]",
    tensor: layerNorm.beta,
    index: 2
  }
];

console.log("======================================");
console.log("LAYERNORM GRADIENT CHECK");
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

  if (!passed) allPassed = false;

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
