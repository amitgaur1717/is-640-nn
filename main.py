from engine import Value
from nn import MLP

# Sample input data (features) and target values
xs = [[2.0, 3.0, -1.0], 
      [3.0, -1.0, 0.5], 
      [0.5, 1.0, 1.0], 
      [1.0, 1.0, -1.0]]  # Input features

ys = [0.0, 1.0, 1.0, 0.0]  # Target values

# Initialize the MLP model with 3 input neurons, 2 hidden layers (4 neurons each), and 1 output neuron
n = MLP(3, [4, 4, 1])

# Training loop over a fixed number of epochs
for k in range(20):
    # Forward pass: generate predictions for each input sample
    ypred = [n(x) for x in xs]
    # Compute the loss as the sum of squared errors between predictions and targets
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Reset gradients for all model parameters before backpropagation
    for p in n.parameters():
        p.grad = 0.0
    # Backpropagate the loss to compute gradients for each parameter
    loss.backward()

    # Update model parameters using gradient descent
    for p in n.parameters():
        p.data += -0.1 * p.grad 

    # Print the current epoch number and loss value
    print(k, loss.data)
