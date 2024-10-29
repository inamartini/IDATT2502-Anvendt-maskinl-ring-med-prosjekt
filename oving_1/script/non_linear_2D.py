import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from a CSV file into a Pandas DataFrame
data = pd.read_csv('../data/day_head_circumference.csv', dtype='float')

# Separate the target variable 'head circumference' from the features
data_y = data.pop('head circumference')

# Convert the remaining data (features) and the target variable to PyTorch tensors
data_x = torch.tensor(data.to_numpy(), dtype=torch.float)
data_y = torch.tensor(data_y.to_numpy(), dtype=torch.float).reshape(-1, 1)

# Define a custom class for a non-linear regression model
class NonLinearRegressionModel:
    def __init__(self, max):
        # Initialize the model's parameters (weights and biases)
        self.max = max
        self.W = torch.tensor([[0.0]], requires_grad=True)  # Weight parameter
        self.b = torch.tensor([[0.0]], requires_grad=True)  # Bias parameter

    # Define the forward function that applies the non-linear transformation
    def f(self, x):
        # The model uses a sigmoid activation function with a scaling factor and offset
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    # Define the loss function, which calculates the mean squared error between predictions and actual values
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

# Instantiate the model using the number of samples as a parameter
model = NonLinearRegressionModel(data.shape[0])

# Set up the optimizer for training, using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)

# Train the model over 100,000 epochs
for epoch in range(100000):
    # Compute the loss and perform backpropagation
    model.loss(data_x, data_y).backward()
    optimizer.step()  # Update the model parameters

    optimizer.zero_grad()  # Reset the gradients for the next iteration

# Print the final values of the model's parameters and the loss after training
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(data_x, data_y)))

# Plot the data and the learned regression function
plt.figure('Non-linear regression in 2D')
plt.title('Head circumference as a function of age')
plt.xlabel('x')
plt.ylabel('y')

# Plot the original data points
plt.scatter(data_x, data_y)

# Generate a range of x values for plotting the regression curve
x = torch.arange(torch.min(data_x), torch.max(data_x), 1.0).reshape(-1, 1)

# Compute the model's predictions for the generated x values
y = model.f(x).detach()

# Plot the regression curve on top of the data points
plt.plot(x, y, color='purple',
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \\frac{1}{1+e^{-z}}$')

plt.legend()
plt.show()