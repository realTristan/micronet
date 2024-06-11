from typing import List, Union
from engine import Node


# Mean Squared Error Loss
def mse_loss(
        y_true: Union[List["Node"], List[int], List[float]],
        y_pred: Union[List["Node"], List[int], List[float]]) -> Union["Node", int, float]:
    """
    Compute the mean squared error loss.

    Args:
        y_true: the true labels
        y_pred: the predicted labels

    Returns:
        the mean squared error loss
    """
    n = len(y_true)

    # compute the sum of the squared differences
    loss = sum([(y_pred[i] - y_true[i]) ** 2 for i in range(n)])

    # return the average loss
    return loss / n
