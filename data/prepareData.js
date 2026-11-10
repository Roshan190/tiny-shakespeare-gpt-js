const fs = require("fs");
const path = require("path");

function prepareData(
  encodeText,
  validationRatio = 0.1,
  sequenceLength = 256
) {
  const filePath = path.join(__dirname, "shakespeare.txt");

  // 1. Read the complete Shakespeare corpus.
  const text = fs.readFileSync(filePath, "utf8");

  if (text.length === 0) {
    throw new Error("Shakespeare dataset is empty.");
  }

  // 2. Convert characters into token IDs.
  const tokenIds = encodeText(text);

  if (!Array.isArray(tokenIds) || tokenIds.length === 0) {
    throw new Error("Tokenizer must return a non-empty array of token IDs.");
  }

  // 3. Split token IDs into training and validation portions.
  const splitIndex = Math.floor(
    tokenIds.length * (1 - validationRatio)
  );

  const trainIds = tokenIds.slice(0, splitIndex);
  const validationIds = tokenIds.slice(splitIndex);

  // 4. Ensure both portions can produce at least one sequence.
  if (trainIds.length <= sequenceLength) {
    throw new Error("Training data is too small for this sequence length.");
  }

  if (validationIds.length <= sequenceLength) {
    throw new Error("Validation data is too small for this sequence length.");
  }

  return {
    text,
    tokenIds,
    trainIds,
    validationIds
  };
}

module.exports = { prepareData };