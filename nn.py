import random
from engine import Node
from typing import List, Union


class Neuron:
    def __init__(self, weights: List[Node], bias: Node):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs: Union[List[Node], List[float], List[int]]) -> Node:
        # compute the sum of the weighted inputs
        weighted_sum = self.bias
        for w, x in zip(self.weights, inputs):
            weighted_sum += w * x

        # apply the activation function
        return weighted_sum.tanh()

    def parameters(self) -> List[Node]:
        return self.weights + [self.bias]

    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias})"


class Layer:
    def __init__(self, neurons: List[Neuron]):
        self.neurons = neurons

    def forward(self, inputs: Union[List[Node], List[float], List[int]]) -> List[Node]:
        return [n.forward(inputs) for n in self.neurons]

    def parameters(self) -> List[Node]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(neurons={self.neurons})"


class MLP:
    def __init__(self, input_size: int, layer_sizes: List[int]):
        # create the hidden layers
        sizes = [input_size] + layer_sizes
        self.layers = []

        for i in range(1, len(sizes)):
            # create the weights and bias for each neuron
            weights = [Node(random.uniform(-1, 1)) for _ in range(sizes[i - 1])]
            bias = Node(random.uniform(-1, 1))

            neurons = [Neuron(weights, bias) for _ in range(sizes[i])]
            self.layers.append(Layer(neurons))

    def forward(self, inputs: Union[List[Node], List[float], List[int]]) -> List[Node]:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def parameters(self) -> List[Node]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def step(self, lr: Union[float, int]) -> None:
        for p in self.parameters():
            p.data += -lr * p.grad

    def __call__(self, inputs: Union[List[Node], List[float], List[int]]) -> List[Node]:
        return self.forward(inputs)

    def __repr__(self):
        return f"MLP(layers={self.layers})"


# execute the code
if __name__ == "__main__":
    from loss import mse_loss

    model = MLP(3, [4, 4, 1])

    lr = 0.1
    epochs = 20

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    for epoch in range(epochs):
        # forward pass
        y_pred = [model(x)[0] for x in xs]
        loss = mse_loss(ys, y_pred)

        # backward pass
        model.zero_grad()
        loss.backward()
        model.step(lr)

        print(epoch, loss.data)
