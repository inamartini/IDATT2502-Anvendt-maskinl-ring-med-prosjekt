import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('../data/length_weight.csv')
data_y = data.pop('weight')
data_x = torch.tensor(data.to_numpy(), dtype=torch.double).reshape(-1, 1)
data_y = torch.tensor(data_y.to_numpy(), dtype=torch.double).reshape(-1, 1)


# Define the model
class LinearRegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)
        self.b = torch.tensor([[0.0]], requires_grad=True, dtype=torch.double)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], 0.001)
for epoch in range(100000):
    model.loss(data_x, data_y).backward()
    optimizer.step()
    optimizer.zero_grad()

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(data_x, data_y)))


plt.plot(data_x, data_y, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(data_x)], [torch.max(data_x)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
