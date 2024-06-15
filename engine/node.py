from typing import Tuple, AnyStr, List, Union, Set
import numpy as np
import random


class Node:
    def __init__(self, data: Union[int, float], label: AnyStr = "", _children: Tuple = (), _op: AnyStr = "") -> None:
        """
        Initialize a node in the computational graph.

        :param data: The data of the node
        :param label: The label of the node
        :param _children: The children of the node
        :param _op: The operation of the node
        """
        # public
        self.data = data
        self.label = label
        self.grad = 0.0

        # private
        self._children = set(_children)
        self._op = _op
        self._backward = lambda: None
        self._coeff = 1.0

    # node in string format
    def __str__(self) -> AnyStr:
        """
        Return the string representation of the node.

        :return: The string representation of the node
        """
        return f"Node(data='{self.data}', grad='{self.grad}', label='{self.label}', op='{self._op}')"

    # node in string format
    def __repr__(self) -> AnyStr:
        return self.__str__()

    # node addition
    def __add__(self, other) -> "Node":
        """
        Add two nodes together.

        :param other: The other node to add
        :return: The sum of the two nodes
        """
        other = other if isinstance(other, Node) else Node(other)

        out = Node(self.data + other.data, _children=(self, other), _op='+')

        def _backward() -> None:
            # consider the numerical coefficient of the variables
            # derivative of 2a with respect to a is 2, etc.
            self.grad += (1 * out.grad) * self._coeff
            other.grad += (1 * out.grad) * other._coeff

            # update the coefficient to account for 2a, or 3a, etc.
            if self == other:
                out._coeff = self._coeff + other._coeff

        out._backward = _backward

        return out

    # node multiplication
    def __mul__(self, other) -> "Node":
        """
        Multiply two nodes.

        :param other: The other node to multiply
        :return: The product of the two nodes
        """
        other = other if isinstance(other, Node) else Node(other)

        out = Node(self.data * other.data, _children=(self, other), _op='*')

        def _backward() -> None:
            # consider the numerical coefficient of the variables
            # derivative of 2a*b with respect to a is 2b, etc.
            self.grad += (other.data * out.grad) * self._coeff
            other.grad += (self.data * out.grad) * other._coeff

            # update the coefficient to account for 2a^2, or 3a^2, etc.
            if self == other:
                out._coeff = self._coeff * other._coeff

        out._backward = _backward

        return out

    # exponentiation
    def __pow__(self, other) -> "Node":
        """
        Exponentiation of the node.

        :param other: What to raise the node to
        :return: The node raised to the power
        """
        # only accept int or float
        assert isinstance(other, (int, float))

        out = Node(self.data ** other, _children=(self,), _op=f'**{other}')

        def _backward() -> None:
            # consider the numerical coefficient of the variables
            # derivative of 2a^b with respect to a is 2b*a^(b-1), etc.
            self.grad += (other * (self.data ** (other - 1)) * out.grad) * self._coeff

            # update the coefficient to account for 2a^2, or 3a^2, etc.
            if self == other:
                out._coeff = self._coeff * other

        out._backward = _backward

        return out

    # right multiplication
    def __rmul__(self, other):
        return self * other

    # division
    def __truediv__(self, other):
        return self * (other ** -1)

    # negation
    def __neg__(self):
        return self * -1

    # subtraction
    def __sub__(self, other):
        return self + (-other)

    # right addition
    def __radd__(self, other):
        return self + other

    # right subtraction
    def __rsub__(self, other):
        return (-self) + other

    # tanh activation function
    def tanh(self) -> "Node":
        """
        Apply the tanh activation function to the node.

        :return: The node with the tanh activation function applied
        """
        out = Node(np.tanh(self.data), _children=(self,), _op='tanh')

        def _backward() -> None:
            # derivative of tanh is 1 - tanh^2
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    # relu activation function
    def relu(self) -> "Node":
        """
        Apply the relu activation function to the node.

        :return: The node with the relu activation function applied
        """
        out = Node(np.maximum(0, self.data), _children=(self,), _op='relu')

        def _backward() -> None:
            # derivative of relu is 1 if x > 0 else 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward

        return out

    # gelu activation function
    def gelu(self) -> "Node":
        """
        Apply the gelu activation function to the node.

        :return: The node with the gelu activation function applied
        """
        out = Node(0.5 * self.data * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3))),
                   _children=(self,), _op='gelu')

        def _backward() -> None:
            # derivative of gelu is very long since product rule is applied
            self.grad += (0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3))) +
                          0.5 * (1 - np.tanh(np.sqrt(2 / np.pi) * (self.data + 0.044715 * self.data ** 3)) ** 2) *
                          (np.sqrt(2 / np.pi) * (1 + 0.134145 * self.data ** 2))) * out.grad

        out._backward = _backward

        return out

    # exponentiation
    def exp(self) -> "Node":
        """
        Raise the node to the power of e (euler's number).

        :return: The node raised to the power of e
        """
        out = Node(np.exp(self.data), _children=(self,), _op=f'exp')

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    # log
    def log(self) -> "Node":
        """
        Apply the log function to the node.

        :return: The node with the log function applied
        """
        out = Node(np.log(self.data), _children=(self,), _op='log')

        def _backward() -> None:
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward

        return out

    # back propagation
    def backward(self) -> None:
        """
        Back propagate the gradient.

        :return: None
        """
        _topo: List["Node"] = []
        _visited: Set["Node"] = set()

        def topo(n: "Node") -> None:
            if n in _visited:
                return

            for child in n._children:
                topo(child)

            _visited.add(n)
            _topo.append(n)

        topo(self)

        self.grad = 1.0
        for node in reversed(_topo):
            node._backward()
