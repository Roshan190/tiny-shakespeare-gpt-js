function getBatch(tokenIds, batchSize, sequenceLength) {
    if (tokenIds.length <= sequenceLength) {
      throw new Error("Not enough tokens to create a training batch.");
    }
  
    const inputIds = [];
    const targetIds = [];
  
    // The last valid starting position for a window of T + 1 tokens.
    const maxStart = tokenIds.length - sequenceLength - 1;
  
    for (let batch = 0; batch < batchSize; batch++) {
      const start = Math.floor(Math.random() * (maxStart + 1));
  
      for (let position = 0; position < sequenceLength; position++) {
        inputIds.push(tokenIds[start + position]);
        targetIds.push(tokenIds[start + position + 1]);
      }
    }
  
    return {
      inputIds,
      targetIds
    };
  }
  
  module.exports = { getBatch };