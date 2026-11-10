const Tensor = require("./Tensor");

function add(a, b) {
    if (a.data.length !== b.data.length) {
        throw new Error("Tensor size must match");
    }

    const outputData = [];

    for (let i = 0; i < a.data.length; i++) {
        outputData.push(a.data[i] + b.data[i]);
    }

    const output = new Tensor(outputData, a.shape);
    output.parent = [a, b];

    output.backward = function () {
        for (let i = 0; i < output.data.length; i++) {
            a.grad[i] += output.grad[i];
            b.grad[i] += output.grad[i];
        }
    }

    return output;
}

function multiply(a, b) {
    if (a.data.length !== b.data.length) {
        throw new Error("Tensor size must match");
    }

    const outputData = [];

    for (let i = 0; i < a.data.length; i++) {
        outputData.push(a.data[i] * b.data[i]);
    }

    const output = new Tensor(outputData, a.shape);
    output.parent = [a, b];

    output.backward = function () {
        for (let i = 0; i < output.data.length; i++) {
            a.grad[i] += output.grad[i] * b.data[i];
            b.grad[i] += output.grad[i] * a.data[i];
        }
    }

    return output;
}

function matmul(a, b) {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
        throw new Error("MatMul currently supports 2D tensors only");
    }

    const rowsA = a.shape[0];
    const colsA = a.shape[1];

    const rowsB = b.shape[0];
    const colsB = b.shape[1];

    if (colsA !== rowsB) {
        throw new Error(`Shape mismatch: [${rowsA}, ${colsA}] × [${rowsB}, ${colsB}]`);
    }

    const outputData = [];

    for (let i = 0; i < rowsA; i++) {
        for (let j = 0; j < colsB; j++) {
            let total = 0;
            for (let k = 0; k < colsA; k++) {
                const aIndex = i * colsA + k;
                const bIndex = k * colsB + j;

                total += a.data[aIndex] * b.data[bIndex];
            }
            outputData.push(total);
        }
    }

    const output = new Tensor(outputData, [rowsA, colsB]);
    output.parent = [a, b];

    output.backward = function () {
        for (let i = 0; i < rowsA; i++) {
            for (let k = 0; k < colsA; k++) {
                let total = 0;
                for (let j = 0; j < colsB; j++) {
                    const outputIndex = i * colsB + j;
                    const bIndex = k * colsB + j;
                    total += output.grad[outputIndex] * b.data[bIndex];
                }
                const aIndex = i * colsA + k;

                a.grad[aIndex] += total;
            }
        }

        for (let k = 0; k < rowsB; k++) {
            for (let j = 0; j < colsB; j++) {
                let total = 0;
                for (let i = 0; i < rowsA; i++) {
                    const aIndex = i * colsA + k;
                    const outputIndex = i * colsB + j;

                    total += a.data[aIndex] * output.grad[outputIndex];
                }

                const bIndex = k * colsB + j;
                b.grad[bIndex] += total;
            }
        }
    }

    return output;
}

function transpose(a) {
    if (a.shape.length !== 2) {
        throw new Error("transpose currently supports 2D tensors");
    }

    const rows = a.shape[0];
    const columns = a.shape[1];

    const outputData = [];

    for (let column = 0; column < columns; column++) {
        for (let row = 0; row < rows; row++) {
            const index = row * columns + column;
            outputData.push(a.data[index]);
        }
    }

    const output = new Tensor(outputData, [columns, rows]);
    output.parent = [a];

    output.backward = function () {
        for (let row = 0; row < rows; row++) {
            for (let column = 0; column < columns; column++) {
                const inputIndex = row * columns + column;
                const outputIndex = column * rows + row;

                a.grad[inputIndex] += output.grad[outputIndex];
            }
        }
    }

    return output;
}

function exp(a) {
    const outputData = [];

    for (let i = 0; i < a.data.length; i++) {
        outputData.push(Math.exp(a.data[i]));
    }

    const output = new Tensor(outputData, a.shape);
    output.parent = [a];

    output.backward = function () {
        for (let i = 0; i < a.data.length; i++) {
            a.grad[i] += output.grad[i] * output.data[i];
        }
    }

    return output;
}

function log(a) {
    const outputData = [];

    for (let i = 0; i < a.data.length; i++) {
        outputData.push(Math.log(a.data[i]));
    }

    const output = new Tensor(outputData, a.shape);
    output.parent = [a];

    output.backward = function () {
        for (let i = 0; i < a.data.length; i++) {
            a.grad[i] += output.grad[i] / a.data[i];
        }
    }

    return output;
}

function sqrt(a) {
    const outputData = [];

    for (let i = 0; i < a.data.length; i++) {
        outputData.push(Math.sqrt(a.data[i]));
    }

    const output = new Tensor(outputData, a.shape);
    output.parent = [a];

    output.backward = function () {
        for (let i = 0; i < a.data.length; i++) {
            a.grad[i] += output.grad[i] / (2 * output.data[i]);
        }
    }

    return output;
}

function forward(x) {
    return linear(x, this.weight, this.bias);
}

function linear(x, weight, bias) {
    if (x.shape.length !== 3) {
        throw new Error("linear expects a 3D input tensor");
    }

    if (weight.shape.length !== 2) {
        throw new Error("weight must be a 2D tensor");
    }

    const batchSize = x.shape[0];
    const sequenceLength = x.shape[1];
    const inputSize = x.shape[2];

    const outputSize = weight.shape[1];

    if (weight.shape[0] !== inputSize) {
        throw new Error("Input size does not match weight");
    }

    if (bias.shape[0] !== outputSize) {
        throw new Error("Bias size does not match output size");
    }

    const outputData = [];

    for (let batch = 0; batch < batchSize; batch++) {
        for (let token = 0; token < sequenceLength; token++) {
            for (let out = 0; out < outputSize; out++) {
                let total = bias.data[out];

                for (let input = 0; input < inputSize; input++) {
                    const xIndex = (batch * sequenceLength * inputSize) + (token * inputSize) + input;
                    const weightIndex = input * outputSize + out;

                    total += x.data[xIndex] * weight.data[weightIndex];
                }

                outputData.push(total);
            }
        }
    }

    const output = new Tensor(outputData, [batchSize, sequenceLength, outputSize]);
    output.parent = [x, weight, bias];

    output.backward = function () {
        for (let batch = 0; batch < batchSize; batch++) {
            for (
                let token = 0;
                token < sequenceLength;
                token++
            ) {
                for (
                    let out = 0;
                    out < outputSize;
                    out++
                ) {
                    const outputIndex =
                        batch *
                        sequenceLength *
                        outputSize +
                        token *
                        outputSize +
                        out;

                    const outputGradient =
                        output.grad[outputIndex];

                    // Bias gradient
                    bias.grad[out] += outputGradient;

                    for (
                        let input = 0;
                        input < inputSize;
                        input++
                    ) {
                        const xIndex =
                            batch *
                            sequenceLength *
                            inputSize +
                            token *
                            inputSize +
                            input;

                        const weightIndex =
                            input *
                            outputSize +
                            out;

                        // dX
                        x.grad[xIndex] +=
                            outputGradient *
                            weight.data[weightIndex];

                        // dW
                        weight.grad[weightIndex] +=
                            outputGradient *
                            x.data[xIndex];
                    }
                }
            }
        }
    };

    return output;
}

module.exports = {
    add,
    multiply,
    matmul,
    transpose,
    exp,
    log,
    sqrt,
    linear,
    forward
}