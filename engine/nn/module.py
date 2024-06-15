from typing import List, Union, Any
from engine.node import Node
from engine.nn.linear import Linear


class Module:
    def __init__(self, sequence: List[Any]) -> None:
        self._sequence = sequence

    # get the parameters of the module
    def parameters(self, _attribute_error_callback: Any = None) -> List[Node]:
        """
        Get the parameters of the module.

        :param _attribute_error_callback: If an attribute error occurs, call this function

        :return:
            the parameters of the module
        """
        params: List[Node] = []

        for row in self._sequence:
            try:
                params.extend(row.parameters())
            except AttributeError:
                if _attribute_error_callback is not None:
                    _attribute_error_callback(row)

        return params

    # zero the gradients of the parameters
    def zero_grad(self) -> None:
        """
        Zero the gradients of the parameters.

        :return:
            None
        """
        for p in self.parameters():
            p.grad = 0.0

    # update the parameters of the module
    def update(self, lr: Union[float, int]) -> None:
        """
        Update the parameters of the module.

        :param lr: The learning rate

        :return:
            None
        """
        for p in self.parameters():
            p.data += -lr * p.grad

    # forward pass through the module
    def forward(self, x: List[Union["Node", int, float]]) -> List[Node]:
        """
        Forward pass through the module.

        :param x: The input to the module

        :return:
            The output after passing through the module sequence
        """
        for row in self._sequence:
            x = row.forward(x)

        return x

    # call the forward method
    def __call__(self, x: List[Union["Node", int, float]]) -> Union[Node, List[Node]]:
        """
        Call the forward method.

        :param x: The input to the module

        :return:
            The output after passing through the module sequence
        """
        return self.forward(x)

    # string representation of the module
    def __str__(self) -> str:
        """
        Return the string representation of the module.

        :return:
            The string representation of the module
        """
        return f"MLP(layers={[str(row) for row in self._sequence]})"

    # string representation of the module
    def __repr__(self) -> str:
        """
        Return the string representation of the module.

        :return:
            The string representation of the module
        """
        return self.__str__()
