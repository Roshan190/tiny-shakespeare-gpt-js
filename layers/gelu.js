
const Tensor = require("../tensor/Tensor");

function gelu(x) {
  const outputData = [];

  // Constants for the tanh approximation of GELU.
  const sqrtTwoOverPi = Math.sqrt(2 / Math.PI);
  const coefficient = 0.044715;

  for (let i = 0; i < x.data.length; i++) {
    const value = x.data[i];

    const inner = sqrtTwoOverPi *
      (value + coefficient * value * value * value);

    const tanhValue = Math.tanh(inner);

    const result = 0.5 * value * (1 + tanhValue);

    outputData.push(result);
  }

  const output = new Tensor(outputData, x.shape);

  output.parent = [x];

  output.backward = () => {
    for (let i = 0; i < x.data.length; i++) {
      const value = x.data[i];

      const inner = sqrtTwoOverPi *
        (value + coefficient * value * value * value);

      const tanhValue = Math.tanh(inner);

      // Derivative of the tanh approximation.
      const innerDerivative = sqrtTwoOverPi *
        (1 + 3 * coefficient * value * value);

      const derivative =
        0.5 * (1 + tanhValue) +
        0.5 * value *
        (1 - tanhValue * tanhValue) *
        innerDerivative;

      x.grad[i] += output.grad[i] * derivative;
    }
  };

  return output;
}

module.exports = gelu;
