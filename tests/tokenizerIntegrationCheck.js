const {
    loadText,
    createTokenizer
  } = require("../tokenizer/tokenizer");
  
  const { prepareData } = require("../data/prepareData");
  
  const config = require("../config");
  
  // 1. Load Shakespeare text.
  const text = loadText("./68-shakespeare-gpt/data/shakespeare.txt");
  
  // 2. Create the actual tokenizer.
  const tokenizer = createTokenizer(text);
  
  // 3. Encode and decode a sample.
  const sample = "The cat";
  const encoded = tokenizer.encode(sample);
  const decoded = tokenizer.decode(encoded);
  
  // 4. Verify round-trip conversion.
  if (decoded !== sample) {
    throw new Error("Tokenizer round-trip test failed.");
  }
  
  // 5. Verify every token ID is valid.
  for (const id of encoded) {
    if (!Number.isInteger(id) || id < 0 || id >= tokenizer.vocabulary.length) {
      throw new Error(`Invalid token ID: ${id}`);
    }
  }
  
  // 6. Prepare the actual Shakespeare dataset.
  const data = prepareData(
    tokenizer.encode,
    0.1,
    config.maxSequenceLength
  );
  
  // 7. Verify the model vocabulary size.
  const vocabSize = tokenizer.vocabulary.length;
  
  console.log("\nTOKENIZER INTEGRATION CHECK");
  console.log("---------------------------");
  console.log("Text characters:", text.length);
  console.log("Vocabulary size:", vocabSize);
  console.log("Sample text:", sample);
  console.log("Encoded:", encoded);
  console.log("Decoded:", decoded);
  
  console.log("\nDATASET CHECK");
  console.log("-------------");
  console.log("Training tokens:", data.trainIds.length);
  console.log("Validation tokens:", data.validationIds.length);
  
  console.log("\nMODEL CONFIGURATION");
  console.log("-------------------");
  console.log("Vocabulary size:", vocabSize);
  console.log("Embedding dimension:", config.dModel);
  console.log("Transformer layers:", config.numLayers);
  console.log("Attention heads:", config.numHeads);
  
  console.log("\nPASS: Tokenizer integration successful.");