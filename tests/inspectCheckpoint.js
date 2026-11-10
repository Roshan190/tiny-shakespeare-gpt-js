const fs = require("fs");

const checkpointPath =
  "./68-shakespeare-gpt/checkpoints/shakespeare-model.json";

// Read the checkpoint.
const checkpoint = JSON.parse(
  fs.readFileSync(checkpointPath, "utf-8")
);

console.log("GPT CHECKPOINT INSPECTION");
console.log("=========================");

console.log("\nMODEL CONFIGURATION");
console.log(checkpoint.config);

// Inspect parameters.
console.log("\nPARAMETER DETAILS");
console.log("-----------------");

let totalParameters = 0;

for (let i = 0; i < checkpoint.parameters.length; i++) {
  const parameter = checkpoint.parameters[i];

  let parameterCount = 1;

  for (const dimension of parameter.shape) {
    parameterCount *= dimension;
  }

  totalParameters += parameterCount;

  console.log(`\nParameter ${i}`);
  console.log("Shape:", parameter.shape);
  console.log("Number of values:", parameterCount);
  console.log("First 5 values:", parameter.data.slice(0, 5));
}

console.log("\nMODEL SUMMARY");
console.log("-------------");
console.log("Total parameter tensors:", checkpoint.parameters.length);
console.log("Total trainable values:", totalParameters);