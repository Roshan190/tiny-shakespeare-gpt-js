const Tensor = require("../tensor/Tensor");
const { randomNormal } = require("../tensor/random");

class PositionalEmbedding {
  constructor(maxSequenceLength, dModel) {
    this.maxSequenceLength = maxSequenceLength;
    this.dModel = dModel;

    const data = [];

    for (let i = 0; i < maxSequenceLength * dModel; i++) {
      data.push(randomNormal() * 0.02);
    }

    this.weight = new Tensor(
      data,
      [maxSequenceLength, dModel]
    );
  }

  forward(batchSize, sequenceLength) {
    if (
      sequenceLength < 1 ||
      sequenceLength > this.maxSequenceLength
    ) {
      throw new Error("Invalid sequence length");
    }

    const outputData = [];

    for (let batch = 0; batch < batchSize; batch++) {
      for (let position = 0; position < sequenceLength; position++) {
        const start = position * this.dModel;

        for (let feature = 0; feature < this.dModel; feature++) {
          outputData.push(this.weight.data[start + feature]);
        }
      }
    }

    const output = new Tensor(
      outputData,
      [batchSize, sequenceLength, this.dModel]
    );

    output.parent = [this.weight];

    output.backward = () => {
      for (let batch = 0; batch < batchSize; batch++) {
        for (let position = 0; position < sequenceLength; position++) {
          const weightStart = position * this.dModel;

          const outputStart =
            (batch * sequenceLength + position) * this.dModel;

          for (let feature = 0; feature < this.dModel; feature++) {
            this.weight.grad[weightStart + feature] +=
              output.grad[outputStart + feature];
          }
        }
      }
    };

    return output;
  }

  parameters() {
    return [this.weight];
  }
}

module.exports = PositionalEmbedding;