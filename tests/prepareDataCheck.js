const { prepareData } = require("../data/prepareData");

// Temporary encoder for testing the data preparation pipeline.
// We'll replace this with your actual character tokenizer.
function testEncoder(text) {
  return Array.from(text).map(character => character.charCodeAt(0));
}

const data = prepareData(testEncoder, 0.1, 256);

console.log("DATASET PREPARATION CHECK");
console.log("-------------------------");

console.log("Original text length:", data.text.length);
console.log("Total token IDs:", data.tokenIds.length);
console.log("Training token IDs:", data.trainIds.length);
console.log("Validation token IDs:", data.validationIds.length);

console.log("\nFirst 20 training IDs:");
console.log(data.trainIds.slice(0, 20));

console.log("\nFirst 20 validation IDs:");
console.log(data.validationIds.slice(0, 20));

console.log("\nPASS: Dataset preparation completed.");