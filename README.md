# micronet
Welcome to micronet!

### What is micronet?
Micronet is a `PyTorch` syntax inspired neural network architecture python library! 
Andrej Karpathy's `micrograd` inspired the actual creation of this library. His video
`The spelled-out intro to neural networks and backpropagation: building micrograd` really
helped me grasp the concepts of neural networks, allowing me to build `micronet` on my own!

### Example
This is an example output from running the `main.py` script.

#### Model Class
Defining our model!

```python3
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
```

#### Model, Criterion, Hyperparameters, & Data
The model initialization, loss function, hyperparameters, and the data that was used!

```python3
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
```

#### Training & Testing Loops
The actual loops used to train and then test the network!

```python3
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
```

#### Terminal Output
The output when all of the above code is put together!

```bash
Test
Prediction: 31.31720183645183, Actual: 30
Prediction: 21.999999999402718, Actual: 22
```