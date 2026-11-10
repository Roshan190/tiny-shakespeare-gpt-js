const Tensor = require("../tensor/Tensor");

class LayerNorm {
  constructor(dModel, epsilon = 1e-5) {
    this.dModel = dModel;
    this.epsilon = epsilon;

    // Trainable scale: initially 1.
    this.gamma = new Tensor(
      new Array(dModel).fill(1),
      [dModel]
    );

    // Trainable offset: initially 0.
    this.beta = new Tensor(
      new Array(dModel).fill(0),
      [dModel]
    );
  }

  forward(x) {
    if (x.shape.length !== 3) {
      throw new Error("LayerNorm expects [batch, sequence, features]");
    }

    const [batchSize, sequenceLength, features] = x.shape;

    if (features !== this.dModel) {
      throw new Error("Input feature size must match dModel");
    }

    const outputData = [];
    const normalized = [];
    const inverseStd = [];

    for (let b = 0; b < batchSize; b++) {
      for (let t = 0; t < sequenceLength; t++) {
        const start = (b * sequenceLength + t) * features;

        // 1. Mean of this token's features.
        let mean = 0;

        for (let i = 0; i < features; i++) {
          mean += x.data[start + i];
        }

        mean /= features;

        // 2. Variance of this token's features.
        let variance = 0;

        for (let i = 0; i < features; i++) {
          const difference = x.data[start + i] - mean;
          variance += difference * difference;
        }

        variance /= features;

        // 3. Normalize and apply gamma/beta.
        const invStd = 1 / Math.sqrt(variance + this.epsilon);
        inverseStd.push(invStd);

        for (let i = 0; i < features; i++) {
          const xHat = (x.data[start + i] - mean) * invStd;

          normalized.push(xHat);

          outputData.push(
            xHat * this.gamma.data[i] + this.beta.data[i]
          );
        }
      }
    }

    const output = new Tensor(outputData, x.shape);

    output.parent = [x, this.gamma, this.beta];

    output.backward = () => {
      const tokenCount = batchSize * sequenceLength;

      for (let token = 0; token < tokenCount; token++) {
        const start = token * features;
        const invStd = inverseStd[token];

        let meanGrad = 0;
        let meanGradTimesXHat = 0;

        // Collect terms needed for the input gradient.
        for (let i = 0; i < features; i++) {
          const dy = output.grad[start + i];
          const xHat = normalized[start + i];

          meanGrad += dy * this.gamma.data[i];
          meanGradTimesXHat += dy * this.gamma.data[i] * xHat;
        }

        meanGrad /= features;
        meanGradTimesXHat /= features;

        for (let i = 0; i < features; i++) {
          const dy = output.grad[start + i];
          const xHat = normalized[start + i];

          // Gradients for shared trainable parameters.
          this.gamma.grad[i] += dy * xHat;
          this.beta.grad[i] += dy;

          // Gradient with respect to the input.
          x.grad[start + i] +=
            invStd *
            (
              dy * this.gamma.data[i] -
              meanGrad -
              xHat * meanGradTimesXHat
            );
        }
      }
    };

    return output;
  }

  parameters() {
    return [this.gamma, this.beta];
  }
}

module.exports = LayerNorm;