import torch
import torchvision

# Hjelp?
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Convolutional layers
        self.pool = torch.nn.MaxPool2d(kernel_size=2) # Max pooling
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2) # 1 input channel, 32 output channels
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2) # 32 input channels, 64 output channels
        self.dense1 = torch.nn.Linear(64 * 7 * 7, 1024) # Fully connected layer
        self.dense2 = torch.nn.Linear(1024, 10) # Fully connected layer

    def logits(self, x):
        x = self.pool(self.conv1(x)) # Apply max pooling after first convolution
        x = self.pool(self.conv2(x)) # Apply max pooling after second convolution
        x = self.dense1(x.view(-1, 64 * 7 * 7)) # Reshape to fit the fully connected layer
        x = self.dense2(x.view(-1, 1024)) # Reshape to fit the fully connected layer
        return x

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print("accuracy = %s" % model.accuracy(x_test, y_test))

# accuracy = tensor(0.9772)
# accuracy = tensor(0.9823)
# accuracy = tensor(0.9837)
# accuracy = tensor(0.9812)
# accuracy = tensor(0.9784)
# accuracy = tensor(0.9724)
# accuracy = tensor(0.9773)
# accuracy = tensor(0.9760)
# accuracy = tensor(0.9813)
# accuracy = tensor(0.9802)
# accuracy = tensor(0.9801)
# accuracy = tensor(0.9803)
# accuracy = tensor(0.9800)
# accuracy = tensor(0.9814)
# accuracy = tensor(0.9796)
# accuracy = tensor(0.9813)
# accuracy = tensor(0.9794)
# accuracy = tensor(0.9845)
# accuracy = tensor(0.9771)
# accuracy = tensor(0.9813)