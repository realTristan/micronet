from engine.node import Node
from typing import List, Union


class ReLU:
    # forward pass through the ReLU activation function
    @staticmethod
    def forward(x: Union[List, Node]) -> Node:
        """
        Forward pass through the ReLU activation function.

        :param x: The input to the ReLU activation function

        :return:
            The output after passing through the ReLU activation function
        """
        if isinstance(x, list):
            out = [xi.relu() for xi in x]
        else:
            out = x.relu()

        return out

    # string representation of the ReLU activation function
    def __str__(self) -> str:
        """
        Return the string representation of the ReLU activation function.

        :return:
            The string representation of the ReLU activation function
        """
        return "ReLU()"

    # string representation of the ReLU activation function
    def __repr__(self) -> str:
        """
        Return the string representation of the ReLU activation function.

        :return:
            The string representation of the ReLU activation function
        """
        return self.__str__()


class Tanh:
    # forward pass through the Tanh activation function
    @staticmethod
    def forward(x: Union[List, Node]) -> Node:
        """
        Forward pass through the Tanh activation function.

        :param x: The input to the Tanh activation function

        :return:
            The output after passing through the Tanh activation function
        """
        if isinstance(x, list):
            out = [xi.tanh() for xi in x]
        else:
            out = x.tanh()

        return out

    # string representation of the Tanh activation function
    def __str__(self) -> str:
        """
        Return the string representation of the Tanh activation function.

        :return:
            The string representation of the Tanh activation function
        """
        return "Tanh()"

    # string representation of the Tanh activation function
    def __repr__(self) -> str:
        """
        Return the string representation of the Tanh activation function.

        :return:
            The string representation of the Tanh activation function
        """
        return self.__str__()
