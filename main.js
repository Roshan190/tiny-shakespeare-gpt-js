const path = require("path");
const config = require("./config");

const GPTModel = require("./models/gptModel");

const {
  loadText,
  createTokenizer
} = require("./tokenizer/tokenizer");

const { prepareData } = require("./data/prepareData");

const { trainModel } = require("./training/trainLoop");

const { saveCheckpoint } = require("./training/checkpoint");

const { saveMetrics } = require("./visualization/saveMetrics");

function main() {
  console.log("====================================");
  console.log("       TINY SHAKESPEARE GPT");
  console.log("====================================");

  // 1. Load Shakespeare dataset.
  console.log("\nLoading Shakespeare dataset...");

  const text = loadText(
    path.join(__dirname, "data", "shakespeare.txt")
  );

  // 2. Build tokenizer.
  console.log("Building tokenizer...");

  const tokenizer = createTokenizer(text);

  const modelConfig = {
    ...config,
    vocabSize: tokenizer.vocabulary.length
  };

  console.log("Vocabulary size:", modelConfig.vocabSize);

  // 3. Prepare training and validation datasets.
  console.log("\nPreparing dataset...");

  const data = prepareData(
    tokenizer.encode,
    modelConfig.validationRatio,
    modelConfig.maxSequenceLength
  );

  console.log("Total tokens:", data.tokenIds.length);
  console.log("Training tokens:", data.trainIds.length);
  console.log("Validation tokens:", data.validationIds.length);

  // 4. Initialize GPT.
  console.log("\nInitializing GPT model...");

  const model = new GPTModel(modelConfig);

  console.log(
    "Trainable parameters:",
    model.parameters().reduce(
      (total, parameter) => total + parameter.data.length,
      0
    )
  );

  // 5. Train GPT.
  console.log("\nStarting training...");

  const history = trainModel(
    model,
    data,
    modelConfig
  );

  // 6. Save trained weights.
  console.log("\nSaving model checkpoint...");

  saveCheckpoint(
    model,
    modelConfig,
    modelConfig.checkpointPath
  );

  // 7. Save training metrics.
  console.log("\nSaving training metrics...");

  saveMetrics(history);

  console.log("\n====================================");
  console.log("       TRAINING COMPLETED");
  console.log("====================================");
}

try {
  main();
} catch (error) {
  console.error("\nTRAINING FAILED");
  console.error(error);
  process.exitCode = 1;
}