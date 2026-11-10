const { getBatch } = require("../data/nextTokenDataset");
const { trainStep } = require("./trainStep");
const { softmaxCrossEntropy } = require("./loss");

function evaluateLoss(
  model,
  validationIds,
  batchSize,
  sequenceLength,
  validationSteps
) {
  let totalLoss = 0;

  for (let step = 0; step < validationSteps; step++) {
    const batch = getBatch(
      validationIds,
      batchSize,
      sequenceLength
    );

    // Forward pass only. No backward pass and no weight update.
    const logits = model.forward(
      batch.inputIds,
      batchSize,
      sequenceLength
    );

    const loss = softmaxCrossEntropy(
      logits,
      batch.targetIds
    );

    totalLoss += loss.data[0];
  }

  return totalLoss / validationSteps;
}

function trainModel(model, data, config, onProgress) {
  const history = [];

  const report = (progress) => {
    if (typeof onProgress === "function") {
      onProgress(progress);
    }
  };

  for (let epoch = 1; epoch <= config.epochs; epoch++) {
    let totalTrainingLoss = 0;

    console.log(`\n========== EPOCH ${epoch} ==========`);

    for (
      let step = 1;
      step <= config.stepsPerEpoch;
      step++
    ) {
      const batch = getBatch(
        data.trainIds,
        config.batchSize,
        config.maxSequenceLength
      );

      const loss = trainStep(
        model,
        batch,
        config.batchSize,
        config.maxSequenceLength,
        config.learningRate
      );

      if (!Number.isFinite(loss)) {
        throw new Error(
          `Invalid training loss at epoch ${epoch}, step ${step}`
        );
      }

      totalTrainingLoss += loss;

      if (step % 10 === 0) {
        console.log(
          `Step ${step}/${config.stepsPerEpoch} | Loss: ${loss.toFixed(4)}`
        );
      }

      report({
        phase: "step",
        epoch,
        totalEpochs: config.epochs,
        step,
        totalSteps: config.stepsPerEpoch,
        trainingLoss: loss,
        validationLoss: null,
        learningRate: config.learningRate
      });
    }

    const averageTrainingLoss =
      totalTrainingLoss / config.stepsPerEpoch;

    const validationLoss = evaluateLoss(
      model,
      data.validationIds,
      config.batchSize,
      config.maxSequenceLength,
      config.validationSteps
    );

    if (!Number.isFinite(validationLoss)) {
      throw new Error(`Invalid validation loss at epoch ${epoch}`);
    }

    const metrics = {
      epoch,
      trainingLoss: averageTrainingLoss,
      validationLoss,
      learningRate: config.learningRate
    };

    history.push(metrics);

    report({
      phase: "epoch",
      epoch,
      totalEpochs: config.epochs,
      step: config.stepsPerEpoch,
      totalSteps: config.stepsPerEpoch,
      trainingLoss: averageTrainingLoss,
      validationLoss,
      learningRate: config.learningRate
    });

    console.log("\nEpoch summary:");
    console.log("Training loss:", averageTrainingLoss.toFixed(4));
    console.log("Validation loss:", validationLoss.toFixed(4));
  }

  return history;
}

module.exports = {
  trainModel,
  evaluateLoss
};
