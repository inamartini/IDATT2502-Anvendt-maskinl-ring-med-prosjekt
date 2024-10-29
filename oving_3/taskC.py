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
        self.pool = torch.nn.MaxPool2d(kernel_size=2)  # Max pooling layer
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)  # First convolutional layer
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Second convolutional layer

        # Fully connected layers
        self.dense1 = torch.nn.Linear(64 * 7 * 7, 1024)  # First fully connected layer
        self.dense2 = torch.nn.Linear(1024, 10)  # Second fully connected layer

        # Dropout layer (for regularization)
        # Dropout randomly drops connections during training to improve generalization
        # Added a Dropout layer with a probability of 50% to prevent overfitting.
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout probability of 50%

        # ReLU activation function
        # ReLU helps the model learn non-linear representations
        # will apply ReLu after each convolutional and dense layer except the output layer
        self.relu = torch.nn.ReLU()

    def logits(self, x):
        # Apply convolutions, pooling, ReLU, and dropout
        x = self.relu(self.pool(self.conv1(x)))  # Convolution -> ReLU -> Pooling
        x = self.relu(self.pool(self.conv2(x)))  # Convolution -> ReLU -> Pooling

        # Flatten and pass through fully connected layers
        x = self.relu(self.dense1(x.view(-1, 64 * 7 * 7)))  # Fully connected -> ReLU
        x = self.dropout(x)  # Apply dropout after first fully connected layer
        x = self.dense2(x)  # Output layer (logits)
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

# Instantiate the extended model
model = ConvolutionalNeuralNetworkModel()

# Optimize using Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        # Compute loss and gradients
        loss = model.loss(x_train_batches[batch], y_train_batches[batch])
        loss.backward()

        # Update model parameters
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients for the next step

    # Print accuracy after each epoch
    print("accuracy = %s" % model.accuracy(x_test, y_test))

# accuracy = tensor(0.9766)
# accuracy = tensor(0.9836)
# accuracy = tensor(0.9869)
# accuracy = tensor(0.9871)
# accuracy = tensor(0.9876)
# accuracy = tensor(0.9839)
# accuracy = tensor(0.9893)
# accuracy = tensor(0.9855)
# accuracy = tensor(0.9899)
# accuracy = tensor(0.9894)
# accuracy = tensor(0.9889)
# accuracy = tensor(0.9876)
# accuracy = tensor(0.9884)
# accuracy = tensor(0.9902)
# accuracy = tensor(0.9902)
# accuracy = tensor(0.9919)
# accuracy = tensor(0.9923)
# accuracy = tensor(0.9909)
# accuracy = tensor(0.9897)
# accuracy = tensor(0.9925)