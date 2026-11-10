
const Tensor = require("../tensor/Tensor");
const causalAttention = require("../layers/causalAttention");

// Q and K are zero, so all allowed positions
// receive equal attention weights.
const q = new Tensor(
  [0, 0, 0, 0, 0, 0],
  [1, 3, 2]
);

const k = new Tensor(
  [0, 0, 0, 0, 0, 0],
  [1, 3, 2]
);

// Three tokens, two features per token.
const v = new Tensor(
  [1, 10, 3, 20, 5, 30],
  [1, 3, 2]
);

const output = causalAttention(q, k, v);

console.log("======================================");
console.log("CAUSAL ATTENTION CHECK");
console.log("======================================");

console.log("Q shape:", q.shape);
console.log("K shape:", k.shape);
console.log("V shape:", v.shape);
console.log("Output shape:", output.shape);

console.log("Output values:");

for (let t = 0; t < 3; t++) {
  const start = t * 2;

  console.log(
    `Token ${t}:`,
    output.data.slice(start, start + 2)
  );
}

// Expected:
// Token 0: [1, 10]
// Token 1: [2, 15]
// Token 2: [3, 20]

const expected = [
  1, 10,
  2, 15,
  3, 20
];

for (let i = 0; i < expected.length; i++) {
  if (Math.abs(output.data[i] - expected[i]) > 1e-6) {
    throw new Error(`Incorrect output at index ${i}`);
  }
}

// Check gradients.
output.grad.fill(1);

q.grad.fill(0);
k.grad.fill(0);
v.grad.fill(0);

const { backward } = require("../tensor/autograd");

backward(output);

const gradients = [
  ...q.grad,
  ...k.grad,
  ...v.grad
];

if (!gradients.every(Number.isFinite)) {
  throw new Error("Attention produced invalid gradients");
}

console.log("Q gradients:", q.grad);
console.log("K gradients:", k.grad);
console.log("V gradients:", v.grad);

console.log("CAUSAL ATTENTION CHECK PASSED");
