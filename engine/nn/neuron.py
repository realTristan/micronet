from typing import List, Union
from engine.node import Node
import random


class Neuron:
    def __init__(self, features: int) -> None:
        self.weights = [Node(random.uniform(-1, 1)) for _ in range(features)]
        self.bias = Node(random.uniform(-1, 1))

    def forward(self, x: List[Union["Node", int, float]]) -> Node:
        out = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)

        return out

    def parameters(self) -> List[Node]:
        return self.weights + [self.bias]

    def __str__(self) -> str:
        return f"Neuron(weights={[str(w) for w in self.weights]}, bias={str(self.bias)})"

    def __repr__(self) -> str:
        return self.__str__()