const path = require("path");
const express = require("express");

const modelService = require("./services/modelService");
const modelRoutes = require("./routes/modelRoutes");
const trainingRoutes = require("./routes/trainingRoutes");
const { errorHandler } = require("./middleware/errorHandler");

const port = Number(process.env.PORT) || 3000;
const host = process.env.HOST || "127.0.0.1";

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  console.error("PORT must be an integer from 1 to 65535.");
  process.exit(1);
}

const summary = modelService.initialize();

const app = express();

app.disable("x-powered-by");
app.use(express.json({ limit: "32kb" }));

app.use("/api", modelRoutes);
app.use("/api", trainingRoutes);

app.use("/api", (req, res) => {
  res.status(404).json({ error: "API route not found." });
});

app.use(express.static(path.join(__dirname, "public")));

app.use(errorHandler);

const server = app.listen(port, host, () => {
  console.log("Tiny Shakespeare GPT");
  console.log(`Vocabulary size: ${summary.vocabularySize}`);

  if (summary.modelLoaded) {
    console.log(
      `Checkpoint loaded (${summary.parameterCount} parameters).`
    );
  } else if (summary.checkpointExists) {
    console.log("Checkpoint found but not loaded.");
    console.log(summary.checkpointError);
  } else {
    console.log("No checkpoint found. The model is not trained yet.");
  }

  console.log(`Listening on http://${host}:${port}`);
});

server.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    console.error(`Port ${port} is already in use.`);
  } else {
    console.error(error.message);
  }

  process.exit(1);
});
