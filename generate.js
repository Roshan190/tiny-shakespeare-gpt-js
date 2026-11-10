const path = require("path");

const config = require("./config");

const GPTModel = require("./models/gptModel");

const {
  loadText,
  createTokenizer
} = require("./tokenizer/tokenizer");

const {
  readCheckpoint,
  loadCheckpoint
} = require("./training/checkpoint");

const {
  generateText
} = require("./generation/generate");

// Read command-line arguments.
const prompt = process.argv[2] || "ROMEO:";

const maxNewTokens = Number(process.argv[3] || 300);

const temperature = Number(process.argv[4] || 0.8);

function main() {
  console.log("====================================");
  console.log("       TINY SHAKESPEARE GPT");
  console.log("          TEXT GENERATION");
  console.log("====================================");

  // Load original corpus to reconstruct the tokenizer.
  const text = loadText(
    path.join(__dirname, "data", "shakespeare.txt")
  );

  const tokenizer = createTokenizer(text);

  // The checkpoint stores the architecture that was actually trained.
  const saved = readCheckpoint(config.checkpointPath);

  if (saved.config.vocabSize !== tokenizer.vocabulary.length) {
    throw new Error(
      "Checkpoint vocabulary does not match the current tokenizer."
    );
  }

  const model = new GPTModel(saved.config);

  const savedConfig = loadCheckpoint(
    model,
    config.checkpointPath
  );

  if (
    savedConfig.maxSequenceLength !== model.maxSequenceLength
  ) {
    throw new Error(
      "Checkpoint context length does not match the model."
    );
  }

  console.log("\nPrompt:", prompt);
  console.log("Temperature:", temperature);
  console.log("New characters:", maxNewTokens);

  console.log("\nGenerating...\n");

  const output = generateText(
    model,
    tokenizer,
    prompt,
    maxNewTokens,
    temperature
  );

  console.log("====================================");
  console.log(output);
  console.log("====================================");
}

try {
  main();
} catch (error) {
  console.error("\nGENERATION FAILED");
  console.error(error);
  process.exitCode = 1;
}
