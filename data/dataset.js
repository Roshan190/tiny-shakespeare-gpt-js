function createSequencePair(tokenIds = [], start, sequenceLength) {
    const input = tokenIds.slice(start, start + sequenceLength);

    const target = tokenIds.slice(start + 1, start + sequenceLength + 1);

    return { input, target }
}

function createBatch(tokenIds = [], batchSize, sequenceLength) {
    const inputs = [];
    const targets = [];

    for (let batch = 0; batch < batchSize; batch++) {
        const maxStart = tokenIds.length - sequenceLength - 1;
        const start = Math.floor(Math.random() * maxStart);

        const sequence = createSequencePair(tokenIds, start, sequenceLength);
        inputs.push(sequence.input);
        targets.push(sequence.target);
    }

    return {
        inputs, targets
    }
}

module.exports = {
    createSequencePair,
    createBatch
}