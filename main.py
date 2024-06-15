import numpy as np
from engine.nn.module import Module
from engine.nn.layer import Layer
from engine.nn.activation import Tanh


class MLPModel(Module):
    def __init__(self) -> None:
        super(MLPModel, self).__init__([
            Layer(3, 128),
            Tanh(),
            Layer(128, 16),
            Tanh(),
            Layer(16, 1),
            Tanh(),
        ])


# execute the code
if __name__ == "__main__":
    from engine.loss import mse_loss

    model = MLPModel()

    lr = 0.01
    epochs = 1000

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    for epoch in range(epochs):
        ypred = [model(x) for x in xs]

        loss = mse_loss(ys, ypred)

        model.zero_grad()
        loss.backward()
        model.update(lr)

        print(f"Epoch {epoch + 1}: Loss={loss.data}")
