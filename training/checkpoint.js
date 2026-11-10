const fs = require("fs");
const path = require("path");

function readCheckpoint(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Checkpoint not found: ${filePath}`);
  }

  let checkpoint;

  try {
    checkpoint = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  } catch (error) {
    throw new Error("Checkpoint file is not valid JSON.");
  }

  if (
    !checkpoint ||
    typeof checkpoint !== "object" ||
    !checkpoint.config ||
    typeof checkpoint.config !== "object" ||
    !Array.isArray(checkpoint.parameters)
  ) {
    throw new Error("Checkpoint is missing config or parameters.");
  }

  return checkpoint;
}

function saveCheckpoint(model, config, filePath) {
  const parameters = model.parameters();

  const checkpoint = {
    config,
    parameters: parameters.map((parameter) => ({
      shape: parameter.shape,
      data: parameter.data
    }))
  };

  const directory = path.dirname(filePath);

  fs.mkdirSync(directory, { recursive: true });

  // Write aside and rename so a crash cannot replace the last good file.
  const tempPath = `${filePath}.tmp`;

  try {
    fs.writeFileSync(
      tempPath,
      JSON.stringify(checkpoint, null, 2),
      "utf-8"
    );

    fs.renameSync(tempPath, filePath);
  } catch (error) {
    try {
      fs.unlinkSync(tempPath);
    } catch (unlinkError) {
      // The original checkpoint is still in place.
    }

    throw error;
  }

  console.log("Checkpoint saved:", filePath);
}

function loadCheckpoint(model, filePath) {
  const checkpoint = readCheckpoint(filePath);
  const parameters = model.parameters();

  if (parameters.length !== checkpoint.parameters.length) {
    throw new Error("Checkpoint parameter count does not match model.");
  }

  for (let i = 0; i < parameters.length; i++) {
    const current = parameters[i];
    const saved = checkpoint.parameters[i];

    if (!saved || !Array.isArray(saved.shape) || !Array.isArray(saved.data)) {
      throw new Error(`Checkpoint parameter ${i} is invalid.`);
    }

    if (JSON.stringify(current.shape) !== JSON.stringify(saved.shape)) {
      throw new Error(`Shape mismatch at parameter ${i}.`);
    }

    if (current.data.length !== saved.data.length) {
      throw new Error(`Data length mismatch at parameter ${i}.`);
    }

    for (let j = 0; j < current.data.length; j++) {
      if (!Number.isFinite(saved.data[j])) {
        throw new Error(`Invalid weight at parameter ${i}, index ${j}.`);
      }

      current.data[j] = saved.data[j];
    }
  }

  console.log("Checkpoint loaded:", filePath);

  return checkpoint.config;
}

module.exports = {
  readCheckpoint,
  saveCheckpoint,
  loadCheckpoint
};
