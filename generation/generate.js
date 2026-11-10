function softmax(logits, temperature) {
    if (!Number.isFinite(temperature) || temperature <= 0) {
      throw new Error("Temperature must be greater than zero.");
    }
  
    const scaled = logits.map(
      value => value / temperature
    );
  
    const maxLogit = Math.max(...scaled);
  
    const exponentials = scaled.map(
      value => Math.exp(value - maxLogit)
    );
  
    const total = exponentials.reduce(
      (sum, value) => sum + value,
      0
    );
  
    return exponentials.map(
      value => value / total
    );
  }
  
  function sampleToken(probabilities) {
    const randomValue = Math.random();
  
    let cumulativeProbability = 0;
  
    for (let i = 0; i < probabilities.length; i++) {
      cumulativeProbability += probabilities[i];
  
      if (randomValue <= cumulativeProbability) {
        return i;
      }
    }
  
    return probabilities.length - 1;
  }
  
  function generateText(
    model,
    tokenizer,
    prompt,
    maxNewTokens = 300,
    temperature = 0.8
  ) {
    if (typeof prompt !== "string" || prompt.length === 0) {
      throw new Error("Please provide a non-empty prompt.");
    }
  
    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
      throw new Error("maxNewTokens must be a non-negative integer.");
    }
  
    let tokenIds = tokenizer.encode(prompt);
  
    if (tokenIds.some(id => id === undefined)) {
      throw new Error(
        "Prompt contains characters that are not in the tokenizer vocabulary."
      );
    }
  
    let generatedText = prompt;
  
    for (let step = 0; step < maxNewTokens; step++) {
      // Keep only the most recent context window.
      const contextIds = tokenIds.slice(
        -model.maxSequenceLength
      );
  
      const contextLength = contextIds.length;
  
      // GPT expects a flattened batch of token IDs.
      const logits = model.forward(
        contextIds,
        1,
        contextLength
      );
  
      // Select logits for the final position.
      const vocabSize = tokenizer.vocabulary.length;
  
      const lastPositionStart =
        (contextLength - 1) * vocabSize;
  
      const nextTokenLogits = logits.data.slice(
        lastPositionStart,
        lastPositionStart + vocabSize
      );
  
      // Convert logits into probabilities.
      const probabilities = softmax(
        nextTokenLogits,
        temperature
      );
  
      // Sample the next character.
      const nextTokenId = sampleToken(probabilities);
  
      // Append the new character.
      tokenIds.push(nextTokenId);
  
      const nextCharacter = tokenizer.decode([
        nextTokenId
      ]);
  
      generatedText += nextCharacter;
    }
  
    return generatedText;
  }
  
  module.exports = {
    generateText
  };