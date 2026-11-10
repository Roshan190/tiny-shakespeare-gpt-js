const fs = require("fs");

function loadText(path) {
    return fs.readFileSync(path, "utf-8");
}

function buildVocabulary(text) {
    const characters = [];
    for (const character of text) {
        let alreadyExists = false;

        for (const existing of characters) {
            if (existing === character) {
                alreadyExists = true;
                break;
            }
        }

        if (!alreadyExists) {
            characters.push(character);
        }
    }

    characters.sort();

    return characters;
}

function buildTokenToId(vocabulary) {
    const tokenId = {};

    for (let i = 0; i < vocabulary.length; i++) {
        tokenId[vocabulary[i]] = i;
    }

    return tokenId;
}

function buildIdToToken(vocabulary) {
    const idToToken = [];

    for (let i = 0; i < vocabulary.length; i++) {
        idToToken.push(vocabulary[i]);
    }

    return idToToken;
}

function createTokenizer(text) {
    const vocabulary = buildVocabulary(text);

    const tokenToId = buildTokenToId(vocabulary);

    const idToToken = buildIdToToken(vocabulary);

    function encode(text) {
        const ids = [];

        for (const character of text) {
            ids.push(tokenToId[character]);
        }

        return ids;
    }

    function decode(ids) {
        let text = "";

        for (const id of ids) {
            text += idToToken[id];
        }

        return text;
    }

    return {
        vocabulary,
        tokenToId,
        idToToken,
        encode,
        decode
    }
}

module.exports = {
    loadText,
    createTokenizer
};
