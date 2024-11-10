import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten(),
                    download=True,
                    train=is_train)

    mnist_data = []
    mnist_labels = []
    for image, labels in dataset:
        mnist_data.append(image)
        mnist_labels.append(labels)
    return np.array(mnist_data), np.array(mnist_labels)


X, y = download_mnist(True)
X_test, y_test = download_mnist(False)


def normalizeData(data: np.array) -> np.array:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


X_norm = normalizeData(X)
X_test_norm = normalizeData(X_test)


def createEntryArray(value: int) -> np.array:
    entry = [0] * 10
    entry[value] = 1
    return entry


def oneHotEncoding(data: np.array) -> np.array:
    # Create new numpy array.
    oneHotData = []
    # Iterate through each element, and build one hot version.
    for entry in data:
        oneHotData.append(createEntryArray(entry))
    return np.array(oneHotData)


y_oneHot = oneHotEncoding(y)
y_test_oneHot = oneHotEncoding(y_test)


def trainValSplit(X, y, valSize=0.2, randomState=None):
    if randomState is not None:
        np.random.seed(randomState)

    # Total number of samples.
    nSamples = X.shape[0]

    # Total number of validation samples.
    nValSamples = int(nSamples * valSize)

    # Shuffle the indices.
    indices = np.random.permutation(nSamples)

    # Split indices into training and validation.
    valIndices = indices[:nValSamples]
    trainIndices = indices[nValSamples:]

    # Use the indices to split the data.
    X_train, X_val = X[trainIndices], X[valIndices]
    y_train, y_val = y[trainIndices], y[valIndices]

    return X_train, X_val, y_train, y_val


X_train, X_val, y_train, y_val = trainValSplit(X_norm,
                                               y_oneHot,
                                               valSize=0.2,
                                               randomState=58)


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def forward(X, W1, W2, b1, b2):
    Z1 = X.dot(W1) + b1
    A1 = relu(Z1)
    Z2 = A1.dot(W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def relu_derivative(Z):
    return Z > 0


def backprg(X, y, W1, W2, b1, b2, Z1, A1, Z2, A2, l_rate):
    # 1. Output layer gradients.
    # dL/dw2 = dL/dZ2 * dZ2/dW2
    # dL/dw2 = (y_hat - y) * A1
    error = A2 - y
    dW2 = A1.T.dot(error)
    dW2 = dW2 / y.shape[0]
    # dL/db2 = dL/dZ2 * dZ2/db2
    # dL/db2 = (y_hat - y) * 1
    db2 = np.sum(error, axis=0, keepdims=True)
    db2 = db2 / y.shape[0]

    # 2. Hidden layer gradients.
    # dL/dW1 = dL/Z2 * dZ2/A1 * dA1/dZ1 * dZ1/dW1
    # dL/dW1 = (y_hat - y) * W2 * relu_deriv * X
    dZ1 = error.dot(W2.T)
    dW1 = dZ1 * relu_derivative(Z1)
    dW1 = X.T.dot(dW1)
    dW1 = dW1 / y.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / y.shape[0]

    # Update weights and biases.
    W1 -= l_rate * dW1
    b1 -= l_rate * db1
    W2 -= l_rate * dW2
    b2 -= l_rate * db2
    return W1, W2, b1, b2


def accuracy(yHat, y):
    return np.mean(np.argmax(yHat, axis=1) == np.argmax(y, axis=1))


def train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
    inputSize = 784
    hiddenSize = 100
    outputSize = 10
    lRate = 0.1

    # W1 = np.random.randn(inputSize, hiddenSize)
    # W2 = np.random.randn(hiddenSize, outputSize)
    W1 = np.random.normal(0, 0.01, size=(inputSize, hiddenSize))
    W2 = np.random.normal(0, 0.01, size=(hiddenSize, outputSize))

    b1 = np.zeros((1, hiddenSize))
    b2 = np.zeros((1, outputSize))

    # Data for learning rate schedueler.
    bestAcc = 0.0
    acceptedEpochs = 4
    noImprovement = 0
    decay = 0.5
    minLearningRate = 1e-5

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        # Shuffle data
        indices = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]

            # Forward and backward propagation
            Z1, A1, Z2, A2 = forward(X_batch, W1, W2, b1, b2)
            W1, W2, b1, b2 = backprg(X_batch, y_batch, W1, W2, b1, b2, Z1, A1, Z2, A2, lRate)

        # Compute accuracy for both training and validation
        _, _, _, A2_train = forward(X_train, W1, W2, b1, b2)
        _, _, _, A2_val = forward(X_val, W1, W2, b1, b2)
        train_acc = accuracy(A2_train, y_train)
        val_acc = accuracy(A2_val, y_val)
        print(f"Epoch {epoch+1}, Training Accuracy: {train_acc*100:.2f}%, Validation Accuracy: {val_acc*100:.2f}%")

        # Check improvement for lerning rate schedueler.
        if val_acc > bestAcc:
            bestAcc = val_acc
            noImprovement = 0
        else:
            noImprovement += 1

        if noImprovement > acceptedEpochs:
            lRate = lRate * decay
            if lRate < minLearningRate:
                lRate = minLearningRate
            print(f"Reducing learning rate to {lRate}")
            noImprovement = 0
            bestAcc = val_acc

    return W1, W2, b1, b2


W1, W2, b1, b2 = train(X_train, y_train, X_val, y_val)
