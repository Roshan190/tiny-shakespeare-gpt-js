function buildTopologicalOrder(node) {
    const visited = new Set();
    const order = [];

    function visit(current) {
        if (visited.has(current)) {
            return;
        }

        visited.add(current);

        for (const parent of current.parent) {
            visit(parent);
        }

        order.push(current);
    }

    visit(node);

    return order;
}

function backward(loss) {
    const nodes = buildTopologicalOrder(loss);

    loss.grad.fill(1);

    nodes.reverse();

    for (const node of nodes) {
        node.backward();
    }
}

module.exports = {
    backward
}