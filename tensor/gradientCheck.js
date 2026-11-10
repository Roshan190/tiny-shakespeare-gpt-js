function numericalGradient(lossFunction, parameter, index) {
    const epsilon = 0.000001;

    const original = parameter.data[index];

    parameter.data[index] = original + epsilon;

    const lossPlus = lossFunction();

    parameter.data[index] = original - epsilon;

    const lossMinus = lossFunction();

    parameter.data[index] = original;

    return (lossPlus - lossMinus) / (2 * epsilon);
}

module.exports = {
    numericalGradient
}