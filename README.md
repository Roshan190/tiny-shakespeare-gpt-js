# Tiny Shakespeare GPT

A tiny character-level GPT built from scratch in JavaScript and trained on [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). There is no TensorFlow, PyTorch, or Transformers library. Tensors, autograd, attention, and the training loop are ordinary JavaScript arrays and loops.

This is a small next-character model, not a ChatGPT-scale chatbot. It continues a prompt one character at a time.

## How the model works

The corpus is split into characters. A deterministic tokenizer sorts the unique characters and assigns each one an id. The model reads a window of ids and predicts the next id at every position.

Training minimizes softmax cross-entropy with plain stochastic gradient descent. There is no Adam, momentum, or weight decay, so the optimizer has no state beyond the weights themselves. Generation turns the logits at the last position into probabilities, scales them by a temperature, and samples the next character. The context window never grows past the model's maximum sequence length.

## Architecture

The network is a decoder-only Transformer:

1. Token embedding and learned positional embedding.
2. A stack of pre-norm Transformer blocks.
3. Each block has causal multi-head self-attention, a residual connection, a GELU feed-forward network, and a second residual connection.
4. A final layer norm and a linear vocabulary head.

`config.js` describes the model used for a fresh training run:

| Setting | Fresh model (`config.js`) |
| --- | --- |
| Embedding dimension | 64 |
| Layers | 2 |
| Heads | 4 |
| Feed-forward size | 256 |
| Context length | 128 |
| Batch size | 16 |
| Parameters | about 116,673 |

The checkpoint already in `checkpoints/shakespeare-model.json` was trained with a smaller architecture: 32 dimensions, 1 layer, 2 heads, context 64, batch 8, and 14,881 parameters. The server loads that saved configuration. It does not silently rebuild the checkpoint with `config.js`.

A forward and backward step on this machine took about **40 ms** for the saved model and about **730 ms** for the fresh `config.js` model. Ten epochs of 100 steps is roughly a minute when resuming the saved model, and roughly 12 to 15 minutes for a fresh model, plus a short validation pass after each epoch.

## Prerequisites

- Node.js 18 or newer

## Installation

```bash
npm install
```

## Train the model

From the command line, `npm run train` runs `main.js`. That script always creates a new model from `config.js`, trains it, and saves `checkpoints/shakespeare-model.json`.

```bash
npm run train
```

You can also train through the server API or the browser. See the API section below. A server training job runs in a worker thread, so the HTTP server keeps answering other requests.

## Start the server

```bash
npm start
```

The same command is:

```bash
node server/server.js
```

The server binds to `127.0.0.1:3000` unless you set `HOST` and `PORT`:

```bash
HOST=127.0.0.1 PORT=3000 npm start
```

`npm run dev` restarts the process when files change. A restart cancels an in-progress training job.

On startup the server reads `data/shakespeare.txt`, rebuilds the tokenizer, and looks for `checkpoints/shakespeare-model.json`.

- A valid checkpoint is loaded once and reused for generation.
- A missing checkpoint starts the server with no model loaded. Generation is refused until training finishes.
- A corrupt or incompatible checkpoint is reported and is not treated as a trained model.

## Browser

Open [http://127.0.0.1:3000](http://127.0.0.1:3000).

The page has two sections:

- **Text generation.** Enter a prompt, choose how many new characters to sample, set the temperature, and generate. Copy puts the result on the clipboard.
- **Model and training.** Shows whether weights are loaded, the checkpoint state, architecture, parameter count, and the current epoch, step, training loss, and validation loss. Choose **Resume saved weights** or **Fresh random weights**, then start training.

The page listens to training progress with server-sent events. Closing the page does not stop training.

## API

Examples assume the server is on `http://127.0.0.1:3000`.

### GET /api/health

```bash
curl http://127.0.0.1:3000/api/health
```

```json
{
  "status": "ok",
  "modelLoaded": true,
  "checkpointExists": true,
  "checkpointValid": true,
  "checkpointError": null,
  "training": false
}
```

### GET /api/model

Returns architecture and checkpoint status. Weight arrays are not included.

```bash
curl http://127.0.0.1:3000/api/model
```

```json
{
  "loaded": true,
  "architecture": "character-level decoder-only GPT",
  "source": "checkpoint",
  "vocabSize": 65,
  "dModel": 32,
  "numLayers": 1,
  "numHeads": 2,
  "headDim": 16,
  "ffnHidden": 64,
  "maxSequenceLength": 64,
  "parameterCount": 14881,
  "freshTraining": {
    "vocabSize": 65,
    "dModel": 64,
    "numLayers": 2,
    "numHeads": 4,
    "headDim": 16,
    "ffnHidden": 256,
    "maxSequenceLength": 128,
    "parameterCount": 116673,
    "batchSize": 16
  },
  "checkpoint": {
    "exists": true,
    "valid": true,
    "path": "checkpoints/shakespeare-model.json",
    "error": null
  }
}
```

`source` is `"checkpoint"` when those fields describe the loaded weights. It is `"config"` when no weights are loaded, and the fields then describe the fresh-training model. `parameterCount` is `null` until a checkpoint is loaded.

### POST /api/generate

Generation requires a loaded checkpoint.

```bash
curl -X POST http://127.0.0.1:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"ROMEO:","maxNewTokens":300,"temperature":0.8}'
```

```json
{
  "text": "ROMEO: ...",
  "prompt": "ROMEO:",
  "parameters": {
    "maxNewTokens": 300,
    "temperature": 0.8
  }
}
```

Validation:

- `prompt` is a non-empty string of at most 2000 characters, and every character must be in the tokenizer vocabulary.
- `maxNewTokens` is an integer from 0 to 1000.
- `temperature` is a number greater than 0 and at most 5.

If no checkpoint is loaded, the response is `409`. Invalid input is `400`.

While a training job is running, generation still uses the weights that were already loaded. The job trains a separate copy.

### POST /api/train

```bash
curl -X POST http://127.0.0.1:3000/api/train \
  -H "Content-Type: application/json" \
  -d '{"mode":"fresh","epochs":10,"stepsPerEpoch":100,"learningRate":0.0003}'
```

```json
{
  "jobId": "…",
  "status": "running",
  "mode": "fresh",
  "epoch": 0,
  "totalEpochs": 10,
  "step": 0,
  "totalSteps": 100,
  "message": "Fresh training started from new random weights."
}
```

The response status is `202`. A second request while that job is running returns `409`.

`epochs` is an integer from 1 to 100. `stepsPerEpoch` is an integer from 1 to 2000. `learningRate` is greater than 0 and at most 1. Batch size, layer sizes, and context length come from `config.js` for `fresh`, and from the checkpoint for `resume`.

### GET /api/train/status

```bash
curl http://127.0.0.1:3000/api/train/status
```

```json
{
  "status": "running",
  "jobId": "…",
  "mode": "resume",
  "epoch": 2,
  "totalEpochs": 10,
  "step": 40,
  "totalSteps": 100,
  "completedSteps": 140,
  "plannedSteps": 1000,
  "trainingLoss": 2.41,
  "validationLoss": 2.55,
  "startedAt": "2026-11-10T16:00:00.000Z",
  "finishedAt": null,
  "error": null,
  "message": "Epoch 2, step 40 of 100.",
  "training": true
}
```

`status` is `idle`, `running`, `completed`, or `failed`. `trainingLoss` is the latest step loss, then the epoch average when the epoch finishes. `validationLoss` updates at the end of an epoch.

### GET /api/train/events

Server-sent events. Each event is `progress`, `complete`, or `failed`, and the data is the same object as `/api/train/status`.

```bash
curl -N http://127.0.0.1:3000/api/train/events
```

Disconnecting a client does not stop the job.

## Resume training

`mode` has two values:

- `fresh` builds a new model from `config.js` and random weights.
- `resume` loads the checkpoint architecture and weights, then continues training.

Resume is not an exact continuation of the previous optimizer step. The optimizer in `training/optimizer.js` is stateless SGD: it subtracts `learningRate * gradient` and stores nothing else. A resumed run keeps the saved weights and starts optimizer state over. The learning rate, epoch count, and steps per epoch come from the request. The batch size and validation settings come from the checkpoint when they were saved, otherwise from `config.js`.

Resume without a valid checkpoint returns `400`.

## Checkpoints

`training/checkpoint.js` writes one JSON file with the model config and every parameter's shape and data. The server save path is `checkpoints/shakespeare-model.json`.

The file is written to `shakespeare-model.json.tmp` and renamed into place. A failed or interrupted training run does not replace the last valid checkpoint. Startup checks that the vocabulary size matches the tokenizer, that `dModel` is divisible by `numHeads`, and that every tensor shape matches the constructed model. Finite numbers are required. A bad file leaves the server up and `modelLoaded` false.

New checkpoints also store the character vocabulary. Older checkpoints that only store `vocabSize` still load.

Command-line generation uses the same rule. `node generate.js "ROMEO:" 300 0.8` builds the model from the checkpoint config, then loads the weights.

## Known limitations

- The model predicts characters. It does not follow chat instructions, and short training still produces rough text.
- Everything runs on one CPU in pure JavaScript. The fresh `config.js` model is much slower per step than the smaller saved checkpoint.
- Only one training job can run. Generation during that job uses the previously loaded weights, not the weights being updated in the worker.
- Resume does not restore optimizer moments, because SGD does not have any.
- Fresh training and `npm run train` use `config.js`. That architecture is larger than the checkpoint currently in the repo. A successful fresh run replaces the checkpoint.
- The server is meant for local use. It binds to localhost and has no authentication. Do not expose the training endpoints on a public network.
- `npm run dev` restarts on file changes and will abandon the worker.

## Dataset

Training text is the Tiny Shakespeare corpus in `data/shakespeare.txt`. Andrej Karpathy assembled and published that file for character-level language models: [karpathy/char-rnn `input.txt`](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt). The underlying plays are William Shakespeare's and are in the public domain.
