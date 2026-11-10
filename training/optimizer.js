function updateParameters(parameters, learningRate) {
    for (const parameter of parameters) {
        for (let i = 0; i < parameter.data.length; i++) {
            parameter.data[i] -= learningRate * parameter.grad[i];
        }
    }
}

function zeroGradients(parameters) {
    for (const parameter of parameters) {
        parameter.grad.fill(0);
    }
}

module.exports = {
    updateParameters,
    zeroGradients
}