from engine.node import Node
from typing import List, Union


class ReLU:
    @staticmethod
    def forward(x: Union[List, Node]) -> Node:
        if isinstance(x, list):
            out = [xi.relu() for xi in x]

        else:
            out = x.relu()

        return out

    def __str__(self) -> str:
        return "ReLU()"

    def __repr__(self) -> str:
        return self.__str__()


class Tanh:
    @staticmethod
    def forward(x: Union[List, Node]) -> Node:
        if isinstance(x, list):
            out = [xi.tanh() for xi in x]

        else:
            out = x.tanh()

        return out

    def __str__(self) -> str:
        return "Tanh()"

    def __repr__(self) -> str:
        return self.__str__()
