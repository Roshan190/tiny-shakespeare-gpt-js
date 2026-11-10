class Tensor {
    constructor(data, shape) {
        this.data = data;
        this.shape = shape;

        this.grad = new Array(data.length).fill(0);

        this.parent = [];
        this.backward = () => {};
    }
}

module.exports = Tensor;