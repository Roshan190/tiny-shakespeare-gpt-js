const Tensor = require("./tensor/Tensor");

const {
  updateParameters,
  zeroGradients
} = require("./training/optimizer");

const weight = new Tensor(
  [0.8, 0.5, -0.2],
  [3]
);

weight.grad = [0.4, -0.2, 0.1];

const parameters = [weight];
const learningRate = 0.1;

console.log("Before:", weight.data);
console.log("Gradients:", weight.grad);

updateParameters(parameters, learningRate);

console.log("After:", weight.data);

zeroGradients(parameters);

console.log("Gradients after reset:", weight.grad);