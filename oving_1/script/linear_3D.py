import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

# Load the data
data = pd.read_csv('../data/day_length_weight.csv', skiprows=1, names=['day', 'length', 'weight'], dtype='float')


# Extract the 'day' column from the DataFrame as the target variable (data_y) and the remaining columns as features (data_x)
data_y = data.pop('day')
data_x = torch.tensor(data.to_numpy(), dtype=torch.float) # Convert features to a PyTorch tensor
data_y = torch.tensor(data_y.to_numpy(), dtype=torch.float).reshape(-1, 1) # Convert target to a PyTorch tensor and reshape it to be a column vector

# Define a simple linear regression model with input size equal to the number of features and output size 1
model = nn.Linear(data_x.shape[1], data_y.shape[1])

# Define the optimizer (Stochastic Gradient Descent) with a small learning rate
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the loss function (Mean Squared Error)
loss_fn = F.mse_loss


# Function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for _ in range(num_epochs):
        pred = model(data_x)  # Forward pass: compute the model's predictions
        loss = loss_fn(pred, data_y)  # Compute the loss between predictions and actual target values
        print(loss.item())
        loss.backward()  # Backward pass: compute the gradients
        opt.step()  # Update the model parameters using the computed gradients
        opt.zero_grad()  # Reset the gradients to zero for the next iteration

    # Print the model's parameters (weights W and bias b) and the final loss
    W, b = model.parameters()
    print("W = %s, b = %s, loss = %s" % (W.data, b.data, loss_fn(model(data_x), data_y)))

# Train the model for 500,000 epochs
fit(500000, model, loss_fn, opt)

# Prepare the data for 3D plotting
xt = data_x.t()[0]
yt = data_y.t()[1]

# Create a 3D plot
fig = plt.figure('Linear regression in 3D')
ax = fig.add_subplot(projection='3d', title="Days as a function of length and weight")

# Generate a range of x and z values for plotting
x = torch.linspace(int(torch.min(xt).item()), int(torch.max(xt).item()), 1000)
z = torch.linspace(int(torch.min(yt).item()), int(torch.max(xt).item()), 1000)

# Scatter plot of the original data points
ax.scatter(x.numpy(),  data_y.numpy(), z.numpy(),label='$(x^{(i)},y^{(i)}, z^{(i)})$')

# Scatter plot of the model's predictions
ax.scatter(x.numpy(), model.f(data_x).detach().numpy(),z.numpy() , label='$\\hat y = f(x) = xW+b$', color="pink")

# Add a legend to the plot
ax.legend()

# Display the plot
plt.show()

