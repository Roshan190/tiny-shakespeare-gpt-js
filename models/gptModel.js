
const Embedding = require("../layers/embedding");
const PositionalEmbedding = require("../layers/positionalEmbedding");
const TransformerBlock = require("../layers/transformerBlock");
const LayerNorm = require("../layers/layerNorm");
const Linear = require("../layers/linear");

const { add } = require("../tensor/operation");

class GPTModel {
  constructor(config) {
    this.vocabSize = config.vocabSize;
    this.dModel = config.dModel;
    this.maxSequenceLength = config.maxSequenceLength;
    this.numLayers = config.numLayers;

    this.tokenEmbedding = new Embedding(
      this.vocabSize,
      this.dModel
    );

    this.positionEmbedding = new PositionalEmbedding(
      this.maxSequenceLength,
      this.dModel
    );

    this.blocks = [];

    for (let i = 0; i < this.numLayers; i++) {
      this.blocks.push(
        new TransformerBlock(
          this.dModel,
          config.numHeads,
          config.ffnHidden
        )
      );
    }

    this.finalNorm = new LayerNorm(this.dModel);

    this.lmHead = new Linear(
      this.dModel,
      this.vocabSize
    );
  }

  forward(tokenIds, batchSize, sequenceLength) {
    // Validate input dimensions.
    if (
      !Number.isInteger(batchSize) ||
      !Number.isInteger(sequenceLength) ||
      batchSize <= 0 ||
      sequenceLength <= 0
    ) {
      throw new Error("Invalid batch size or sequence length");
    }

    if (batchSize * sequenceLength !== tokenIds.length) {
      throw new Error(
        "Token count must equal batchSize * sequenceLength"
      );
    }

    if (sequenceLength > this.maxSequenceLength) {
      throw new Error(
        "Sequence exceeds maximum sequence length"
      );
    }

    // 1. Convert token IDs into embeddings.
    const tokenEmbeddings = this.tokenEmbedding.forward(
      tokenIds,
      [batchSize, sequenceLength]
    );

    // 2. Generate positional embeddings.
    const positionEmbeddings = this.positionEmbedding.forward(
      batchSize,
      sequenceLength
    );

    // 3. Combine token and positional information.
    let x = add(tokenEmbeddings, positionEmbeddings);

    // 4. Pass through all Transformer blocks.
    for (const block of this.blocks) {
      x = block.forward(x);
    }

    // 5. Final normalization.
    x = this.finalNorm.forward(x);

    // 6. Generate vocabulary logits.
    return this.lmHead.forward(x);
  }

  parameters() {
    const result = [
      ...this.tokenEmbedding.parameters(),
      ...this.positionEmbedding.parameters()
    ];

    for (const block of this.blocks) {
      result.push(...block.parameters());
    }

    result.push(...this.finalNorm.parameters());
    result.push(...this.lmHead.parameters());

    return result;
  }
}

module.exports = GPTModel;
