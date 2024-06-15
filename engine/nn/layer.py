from typing import List, Union
from engine.node import Node
from engine.nn.neuron import Neuron


class Layer:
    def __init__(self, features: int, channels: int) -> None:
        self.neurons = [Neuron(features) for _ in range(channels)]

    def forward(self, x: List[Union["Node", int, float]]) -> Union[Node, List[Node]]:
        out = [n.forward(x) for n in self.neurons]

        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Node]:
        return [p for n in self.neurons for p in n.parameters()]

    def __str__(self) -> str:
        return f"Layer(neurons={[str(n) for n in self.neurons]})"

    def __repr__(self) -> str:
        return self.__str__()

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]
