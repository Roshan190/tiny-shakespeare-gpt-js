
const Tensor = require("../tensor/Tensor");
const MultiHeadAttention = require("../layers/multiHeadAttention");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

const dModel = 4;
const numHeads = 2;

const attention = new MultiHeadAttention(
  dModel,
  numHeads
);

// Make all parameters deterministic and non-identical.
for (let p = 0; p < attention.parameters().length; p++) {
  const parameter = attention.parameters()[p];

  for (let i = 0; i < parameter.data.length; i++) {
    parameter.data[i] =
      Math.sin((p + 1) * (i + 1) * 0.37) * 0.15;
  }
}

// Input: [batch=1, sequence=3, features=4]
const x = new Tensor(
  [
    0.2, 0.5, 0.1, 0.8,
    0.7, 0.3, 0.9, 0.4,
    0.6, 0.2, 0.5, 0.1
  ],
  [1, 3, dModel]
);

// Different coefficients for every output feature.
const coefficients = [
  0.3, -0.7, 0.5, 0.2,
  -0.4, 0.8, 0.6, -0.3,
  0.9, -0.2, 0.4, 0.7
];

// Calculate scalar loss using a fresh forward pass.
function calculateLoss() {
  const output = attention.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// Build the differentiable scalar loss.
function createLoss() {
  const output = attention.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  const loss = new Tensor([total], [1]);

  // Singular property: parent.
  loss.parent = [output];

  loss.backward = () => {
    for (let i = 0; i < output.data.length; i++) {
      output.grad[i] += coefficients[i] * loss.grad[0];
    }
  };

  return loss;
}

// Reset all gradients.
x.grad.fill(0);

for (const parameter of attention.parameters()) {
  parameter.grad.fill(0);
}

// Analytical gradients.
const loss = createLoss();
backward(loss);

// Select representative values from both heads
// and the final output projection.
const tests = [
  {
    name: "Input X[0]",
    tensor: x,
    index: 0
  },
  {
    name: "Head 1 Query weight[0]",
    tensor: attention.heads[0].query.weight,
    index: 0
  },
  {
    name: "Head 1 Key weight[1]",
    tensor: attention.heads[0].key.weight,
    index: 1
  },
  {
    name: "Head 1 Value weight[0]",
    tensor: attention.heads[0].value.weight,
    index: 0
  },
  {
    name: "Head 2 Query weight[1]",
    tensor: attention.heads[1].query.weight,
    index: 1
  },
  {
    name: "Head 2 Key weight[0]",
    tensor: attention.heads[1].key.weight,
    index: 0
  },
  {
    name: "Head 2 Value weight[1]",
    tensor: attention.heads[1].value.weight,
    index: 1
  },
  {
    name: "Output projection weight[0]",
    tensor: attention.outputProjection.weight,
    index: 0
  },
  {
    name: "Output projection bias[0]",
    tensor: attention.outputProjection.bias,
    index: 0
  }
];

console.log("======================================");
console.log("MULTI-HEAD ATTENTION GRADIENT CHECK");
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
