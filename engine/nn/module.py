from typing import List, Union, Any
from engine.node import Node
from engine.nn.layer import Layer


class Module:
    def __init__(self, _sequence: List[Any]) -> None:
        self._sequence = _sequence

    def parameters(self) -> List[Node]:
        # if row is a layer, get the parameters of the layer
        return [p for row in self._sequence if isinstance(row, Layer) for p in row.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def update(self, lr: Union[float, int]) -> None:
        for p in self.parameters():
            p.data += -lr * p.grad

    def forward(self, x: List[Union["Node", int, float]]) -> List[Node]:
        for row in self._sequence:
            x = row.forward(x)

        return x

    def __call__(self, x: List[Union["Node", int, float]]) -> Union[Node, List[Node]]:
        return self.forward(x)

    def __str__(self) -> str:
        return f"MLP(layers={[str(row) for row in self._sequence]})"

    def __repr__(self) -> str:
        return self.__str__()