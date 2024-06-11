from typing import Tuple, AnyStr, List, Union, Set
import numpy as np


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

    # print the node in string form
    def __repr__(self) -> AnyStr:
        """
        Return the string representation of the node.

        :return: The string representation of the node
        """
        return f"Node(data='{self.data}', grad='{self.grad}', label='{self.label}', op='{self._op}')"

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
            # if self and other are the same, return 2.0, else 1.0
            # this is because the derivative of, for ex: a + a (2a) with respect
            # to the output is 2 and not 1
            c = 2.0 if self == other else 1.0

            self.grad += c * out.grad
            other.grad += c * other.grad

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
            # derivative of A * B with respect to A is B and vice versa
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

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
            # derivative of A^B with respect to A is B * A^(B-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    # tanh activation function
    def tanh(self) -> "Node":
        """
        Apply the tanh activation function to the node.

        :return: The node with the tanh activation function applied
        """
        out = Node(np.tanh(self.data), _children=(self,), _op='tanh')

        def _backward() -> None:
            # derivative of tanh is 1 - tanh^2
            self.grad += (1 - np.tanh(self.data) ** 2) * out.grad

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
