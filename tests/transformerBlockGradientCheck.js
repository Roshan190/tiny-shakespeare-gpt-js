
const Tensor = require("../tensor/Tensor");
const TransformerBlock = require("../layers/transformerBlock");
const { backward } = require("../tensor/autograd");
const { numericalGradient } = require("../tensor/gradientCheck");

// Small Transformer configuration.
const dModel = 4;
const numHeads = 2;
const ffnHidden = 8;

const block = new TransformerBlock(
  dModel,
  numHeads,
  ffnHidden
);

// Make initialization deterministic for reproducible tests.
const parameters = block.parameters();

for (let p = 0; p < parameters.length; p++) {
  const parameter = parameters[p];

  for (let i = 0; i < parameter.data.length; i++) {
    parameter.data[i] =
      Math.sin((p + 1) * (i + 1) * 0.37) * 0.08;
  }
}

// LayerNorm scale and shift should start at 1 and 0.
block.norm1.gamma.data.fill(1);
block.norm1.beta.data.fill(0);

block.norm2.gamma.data.fill(1);
block.norm2.beta.data.fill(0);

// Input: [batch=1, sequence=3, features=4]
const x = new Tensor(
  [
    0.2, 0.5, 0.1, 0.8,
    0.7, 0.3, 0.9, 0.4,
    0.6, 0.2, 0.5, 0.1
  ],
  [1, 3, dModel]
);

// Weighted scalar objective.
const coefficients = [
  0.3, -0.7, 0.5, 0.2,
  -0.4, 0.8, 0.6, -0.3,
  0.9, -0.2, 0.4, 0.7
];

// Fresh forward pass for numerical gradient calculation.
function calculateLoss() {
  const output = block.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  return total;
}

// Build the differentiable loss for backpropagation.
function createLoss() {
  const output = block.forward(x);

  let total = 0;

  for (let i = 0; i < output.data.length; i++) {
    total += output.data[i] * coefficients[i];
  }

  const loss = new Tensor([total], [1]);

  // Our autograd convention uses parent.
  loss.parent = [output];

  loss.backward = () => {
    for (let i = 0; i < output.data.length; i++) {
      output.grad[i] += coefficients[i] * loss.grad[0];
    }
  };

  return loss;
}

// Reset all persistent gradients.
x.grad.fill(0);

for (const parameter of parameters) {
  parameter.grad.fill(0);
}

// Calculate analytical gradients.
const loss = createLoss();
backward(loss);

// Select representative parameters from the entire block.
const tests = [
  {
    name: "Input X[0]",
    tensor: x,
    index: 0
  },

  // First LayerNorm.
  {
    name: "Norm1 gamma[0]",
    tensor: block.norm1.gamma,
    index: 0
  },
  {
    name: "Norm1 beta[1]",
    tensor: block.norm1.beta,
    index: 1
  },

  // Attention head 1.
  {
    name: "Head1 Query weight[0]",
    tensor: block.attention.heads[0].query.weight,
    index: 0
  },
  {
    name: "Head1 Key weight[0]",
    tensor: block.attention.heads[0].key.weight,
    index: 0
  },
  {
    name: "Head1 Value weight[0]",
    tensor: block.attention.heads[0].value.weight,
    index: 0
  },

  // Attention head 2.
  {
    name: "Head2 Query weight[0]",
    tensor: block.attention.heads[1].query.weight,
    index: 0
  },
  {
    name: "Head2 Key weight[1]",
    tensor: block.attention.heads[1].key.weight,
    index: 1
  },
  {
    name: "Head2 Value weight[1]",
    tensor: block.attention.heads[1].value.weight,
    index: 1
  },

  // Attention output projection.
  {
    name: "Attention output weight[0]",
    tensor: block.attention.outputProjection.weight,
    index: 0
  },

  // Second LayerNorm.
  {
    name: "Norm2 gamma[0]",
    tensor: block.norm2.gamma,
    index: 0
  },
  {
    name: "Norm2 beta[1]",
    tensor: block.norm2.beta,
    index: 1
  },

  // Feed-forward network.
  {
    name: "FFN Linear1 weight[0]",
    tensor: block.feedForward.linear1.weight,
    index: 0
  },
  {
    name: "FFN Linear1 bias[0]",
    tensor: block.feedForward.linear1.bias,
    index: 0
  },
  {
    name: "FFN Linear2 weight[0]",
    tensor: block.feedForward.linear2.weight,
    index: 0
  },
  {
    name: "FFN Linear2 bias[0]",
    tensor: block.feedForward.linear2.bias,
    index: 0
  }
];

// Compare analytical and numerical gradients.
console.log("======================================");
console.log("TRANSFORMER BLOCK GRADIENT CHECK");
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
