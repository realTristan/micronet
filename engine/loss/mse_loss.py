from engine.node import Node
from typing import List, Union


_ForwardInputType = Union[List[Union[Node, int, float]], Node, int, float]


class MSELoss:
    def __init__(self) -> None:
        self.loss: Union[Node, None] = None

    # call the forward method
    def __call__(self, y_true: _ForwardInputType, y_pred: _ForwardInputType) -> None:
        """
        Call the forward method.

        :param y_true: The true labels
        :param y_pred: The predicted labels

        :return:
            None
        """
        return self.forward(y_true, y_pred)

    # string representation of the loss
    def __str__(self) -> str:
        """
        Return the string representation of the loss.

        :return:
            The string representation of the loss
        """
        return str(self.loss.data if self.loss else None)

    # string representation of the loss
    def __repr__(self) -> str:
        """
        Return the string representation of the loss.

        :return:
            The string representation of the loss
        """
        return self.__str__()

    # backward pass through the loss
    def backward(self) -> None:
        """
        Backward pass through the loss.

        :return:
            None
        """
        return self.loss.backward()

    # forward pass through the loss
    def forward(self, y_true: _ForwardInputType, y_pred: _ForwardInputType) -> None:
        """
        Compute the mean squared error loss.

        Args:
            y_true: the true labels
            y_pred: the predicted labels

        Returns:
            None
        """

        # if type of y_true or y_pred is not list, convert it to list
        if not isinstance(y_true, list):
            y_true = [y_true]

        if not isinstance(y_pred, list):
            y_pred = [y_pred]

        self.loss = sum(((yp - yt) ** 2) for yt, yp in zip(y_true, y_pred))
