const { createRandomTensor } = require("../tensor/random");
const Tensor = require("../tensor/Tensor");
const { linear } = require("../tensor/operation");

class Linear {
    constructor(inputSize, outputSize) {
        const scale = Math.sqrt(2 / inputSize);

        this.weight = createRandomTensor(inputSize, outputSize, scale);

        this.bias = new Tensor(new Array(outputSize).fill(0), [outputSize]);
    }

    forward(x){
        return linear(x, this.weight, this.bias);
    }

    parameters(){
        return [this.weight, this.bias];
    }
}

module.exports = Linear;