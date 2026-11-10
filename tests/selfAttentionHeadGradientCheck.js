
const Tensor = require("../tensor/Tensor");
const SelfAttentionHead = require("../layers/selfAttentionHead");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// Small attention head.
const dModel = 3;
const headDim = 2;

const attentionHead = new SelfAttentionHead(
  dModel,
  headDim
);

// Set deterministic weights so the test is reproducible.
attentionHead.query.weight.data = [
  0.2, -0.1,
  0.4, 0.3,
  -0.2, 0.5
];

attentionHead.query.bias.data = [0.01, -0.02];

attentionHead.key.weight.data = [
  -0.3, 0.2,
  0.1, 0.4,
  0.5, -0.2
];

attentionHead.key.bias.data = [0.03, 0.01];

attentionHead.value.weight.data = [
  0.6, -0.1,
  -0.4, 0.3,
  0.2, 0.5
];

attentionHead.value.bias.data = [-0.01, 0.02];

// Input: [batch=1, sequence=3, features=3]
const x = new Tensor(
  [
    0.2, 0.5, 0.1,
    0.7, 0.3, 0.9,
    0.6, 0.2, 0.5
  ],
  [1, 3, dModel]
);

// Coefficients for our scalar loss.
const coefficients = [
  0.3, -0.7,
  0.5, 0.2,
  -0.4, 0.8
];

// Calculate loss using a fresh forward pass.
function calculateLoss() {
  const output = attentionHead.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// Build a differentiable scalar loss.
function createLoss() {
  const output = attentionHead.forward(x);

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
x.grad.fill(0);

for (const parameter of attentionHead.parameters()) {
  parameter.grad.fill(0);
}

// Analytical gradients.
const loss = createLoss();
backward(loss);

// Select representative parameters from every projection.
const tests = [
  {
    name: "Input X[0]",
    tensor: x,
    index: 0
  },

  {
    name: "Query weight[0]",
    tensor: attentionHead.query.weight,
    index: 0
  },
  {
    name: "Query weight[3]",
    tensor: attentionHead.query.weight,
    index: 3
  },
  {
    name: "Query bias[0]",
    tensor: attentionHead.query.bias,
    index: 0
  },

  {
    name: "Key weight[0]",
    tensor: attentionHead.key.weight,
    index: 0
  },
  {
    name: "Key weight[3]",
    tensor: attentionHead.key.weight,
    index: 3
  },
  {
    name: "Key bias[0]",
    tensor: attentionHead.key.bias,
    index: 0
  },

  {
    name: "Value weight[0]",
    tensor: attentionHead.value.weight,
    index: 0
  },
  {
    name: "Value weight[3]",
    tensor: attentionHead.value.weight,
    index: 3
  },
  {
    name: "Value bias[0]",
    tensor: attentionHead.value.bias,
    index: 0
  }
];

console.log("======================================");
console.log("TRAINABLE ATTENTION GRADIENT CHECK");
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
