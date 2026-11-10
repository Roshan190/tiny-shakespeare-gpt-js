const Tensor = require("../tensor/Tensor");
const { randomNormal } = require("../tensor/random");

class Embedding {
  constructor(vocabSize, dModel) {
    this.vocabSize = vocabSize;
    this.dModel = dModel;

    const data = [];

    for (let i = 0; i < vocabSize * dModel; i++) {
      data.push(randomNormal() * 0.02);
    }

    this.weight = new Tensor(
      data,
      [vocabSize, dModel]
    );
  }

  forward(tokenIds, inputShape = [tokenIds.length]) {
    if (inputShape.reduce((a, b) => a * b, 1) !== tokenIds.length) {
      throw new Error("Input shape does not match token IDs");
    }

    const outputData = [];

    for (const tokenId of tokenIds) {
      if (
        !Number.isInteger(tokenId) ||
        tokenId < 0 ||
        tokenId >= this.vocabSize
      ) {
        throw new Error(`Invalid token ID: ${tokenId}`);
      }

      const start = tokenId * this.dModel;

      for (let feature = 0; feature < this.dModel; feature++) {
        outputData.push(
          this.weight.data[start + feature]
        );
      }
    }

    const outputShape = [...inputShape, this.dModel];

    const output = new Tensor(outputData, outputShape);

    output.parent = [this.weight];

    output.backward = () => {
      for (let position = 0; position < tokenIds.length; position++) {
        const tokenId = tokenIds[position];
        const embeddingStart = tokenId * this.dModel;
        const outputStart = position * this.dModel;

        for (let feature = 0; feature < this.dModel; feature++) {
          this.weight.grad[embeddingStart + feature] +=
            output.grad[outputStart + feature];
        }
      }
    };

    return output;
  }

  parameters() {
    return [this.weight];
  }
}

module.exports = Embedding;