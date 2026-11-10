const Tensor = require("./Tensor");

function randomNormal() {
    let u = 0;
    let v = 0;

    while (u === 0) {
        u = Math.random();
    }

    while (v === 0) {
        v = Math.random()
    }

    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function createRandomTensor(rows, cols, scale) {
    const data = [];

    for (let i = 0; i < rows * cols; i++) {
        data.push(randomNormal() * scale);
    }

    return new Tensor(data, [rows, cols]);
}

module.exports = {
    randomNormal,
    createRandomTensor,
}