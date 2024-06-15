from engine.node import Node
from typing import List, Union


_ForwardInputType = Union[List[Union[Node, int, float]], Node, int, float]


class MSELoss:
    def __init__(self) -> None:
        self.loss: Union[Node, None] = None

    def __call__(self, y_true: _ForwardInputType, y_pred: _ForwardInputType) -> None:
        return self.forward(y_true, y_pred)

    def __str__(self) -> str:
        return str(self.loss.data if self.loss else None)

    def __repr__(self) -> str:
        return self.__str__()

    def backward(self) -> None:
        return self.loss.backward()

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
