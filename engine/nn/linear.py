from typing import List, Union
from engine.node import Node
import random


class Neuron:
    def __init__(self, features: int) -> None:
        self.weights = [Node(random.uniform(-1, 1)) for _ in range(features)]
        self.bias = Node(random.uniform(-1, 1))

    # forward pass through the neuron
    def forward(self, x: List[Union["Node", int, float]]) -> Node:
        """
        Forward pass through the neuron.

        :param x: the input to the neuron
        :return:
            The output after passing through the neuron
        """
        out = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)

        return out

    # return the parameters of the neuron
    def parameters(self) -> List[Node]:
        """
        Return the parameters of the neuron.

        :return:
            The parameters of the neuron
        """
        return self.weights + [self.bias]

    # the string representation of the neuron
    def __str__(self) -> str:
        """
        the string representation of the neuron.

        :return:
            The string representation of the neuron
        """
        return f"Neuron(weights={[str(w) for w in self.weights]}, bias={str(self.bias)})"

    # the string representation of the neuron
    def __repr__(self) -> str:
        """
        the string representation of the neuron.

        :return:
            The string representation of the neuron
        """
        return self.__str__()


class Linear:
    def __init__(self, features: int, channels: int) -> None:
        self.neurons = [Neuron(features) for _ in range(channels)]

    # forward pass through the layer
    def forward(self, x: List[Union["Node", int, float]]) -> Union[Node, List[Node]]:
        """
        Forward pass through the layer.

        :param x: the input to the layer

        :return:
            The output after passing through the layer
        """
        out = [n.forward(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    # return the parameters of the layer
    def parameters(self) -> List[Node]:
        """
        Return the parameters of the layer.

        :return:
            The parameters of the layer
        """
        return [p for n in self.neurons for p in n.parameters()]

    # the string representation of the layer
    def __str__(self) -> str:
        """
        the string representation of the layer.

        :return:
            The string representation of the layer
        """
        return f"Layer(neurons={[str(n) for n in self.neurons]})"

    # the string representation of the layer
    def __repr__(self) -> str:
        """
        the string representation of the layer.

        :return:
            The string representation of the layer
        """
        return self.__str__()

    # return the number of neurons in the layer
    def __len__(self) -> int:
        """
        Return the number of neurons in the layer.

        :return:
            The number of neurons in the layer
        """
        return len(self.neurons)
