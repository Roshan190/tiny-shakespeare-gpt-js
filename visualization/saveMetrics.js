const fs = require("fs");
const path = require("path");

function saveMetrics(history) {
  const directory = path.join(__dirname);

  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory, { recursive: true });
  }

  const filePath = path.join(
    directory,
    "training-metrics.json"
  );

  fs.writeFileSync(
    filePath,
    JSON.stringify(history, null, 2),
    "utf8"
  );

  console.log("\nTraining metrics saved:", filePath);
}

module.exports = { saveMetrics };