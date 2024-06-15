import numpy as np
from engine.nn.module import Module
from engine.nn.linear import Linear
from engine.nn.activation import Tanh, ReLU, GeLU
from engine.loss import MSELoss


class MLPModel(Module):
    def __init__(self) -> None:
        super(MLPModel, self).__init__([
            Linear(3, 128),
            ReLU(),
            Linear(128, 16),
            ReLU(),
            Linear(16, 1),
            ReLU(),
        ])


# execute the code
if __name__ == "__main__":
    ## Model, Criterion, and Hyperparameters
    model = MLPModel()
    criterion = MSELoss()
    lr = 0.1
    epochs = 100

    ## Train data and corresponding labels
    train_data = [
        [1.70, 70, 1],
        [1.60, 50, 0],
        [1.80, 80, 1],
        [1.85, 90, 1],
        [1.75, 75, 0],
        [1.65, 55, 0],
    ]
    train_labels = [25, 20, 30, 35, 27, 22]

    ## Test data and corresponding labels
    test_data = [
        [1.75, 80, 1],
        [1.65, 55, 0],
    ]
    test_labels = [30, 22]

    ##
    ## Train the network
    ##
    for epoch in range(epochs):
        for x, y in zip(train_data, train_labels):
            model.zero_grad()

            y_pred = model(x)

            criterion(y, y_pred)
            criterion.backward()

        model.update(lr)

        print(f"Epoch {epoch}, Loss: {criterion}")

    ##
    ## Test the network
    ##
    print("\nTest")
    for i, x in enumerate(test_data):
        y = model(x)
        print(f"Prediction: {y.data}, Actual: {test_labels[i]}")
