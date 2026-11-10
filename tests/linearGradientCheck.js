const Tensor = require("../tensor/Tensor");
const Linear = require("../layers/linear");
const { meanSquaredError } = require("../training/loss");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// Dataset: y = 2x
const inputs = new Tensor(
  [1, 2, 3],
  [1, 3, 1]
);

const targets = [2, 4, 6];

const model = new Linear(1, 1);

// Use known parameters so the test is reproducible.
model.weight.data[0] = 0.5;
model.bias.data[0] = 0;

const parameters = model.parameters();

// Analytical gradients
const predictions = model.forward(inputs);
const loss = meanSquaredError(predictions, targets);

backward(loss);

const analyticalWeightGradient = model.weight.grad[0];
const analyticalBiasGradient = model.bias.grad[0];

// Numerical gradients
function calculateLoss() {
  const predictions = model.forward(inputs);
  const loss = meanSquaredError(predictions, targets);

  return loss.data[0];
}

const numericalWeightGradient = numericalGradient(
  calculateLoss,
  model.weight,
  0
);

const numericalBiasGradient = numericalGradient(
  calculateLoss,
  model.bias,
  0
);

function checkGradient(name, analytical, numerical) {
  const difference = Math.abs(analytical - numerical);
  const passed = difference < 0.0001;

  console.log(
    `${name}: analytical=${analytical.toFixed(6)}, ` +
    `numerical=${numerical.toFixed(6)}, ` +
    `difference=${difference.toExponential(3)} ` +
    `${passed ? "PASS" : "FAIL"}`
  );

  return passed;
}

console.log("====================================");
console.log("LINEAR GRADIENT CHECK");
console.log("====================================");

const weightPassed = checkGradient(
  "Weight",
  analyticalWeightGradient,
  numericalWeightGradient
);

const biasPassed = checkGradient(
  "Bias",
  analyticalBiasGradient,
  numericalBiasGradient
);

if (!weightPassed || !biasPassed) {
  throw new Error("Gradient check failed");
}

console.log("All gradient checks passed.");