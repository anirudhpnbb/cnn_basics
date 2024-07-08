import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x>0, 1, 0)

input_size = 2
hidden_size=3
output_size=1

np.random.seed(42)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.random.rand(hidden_size)
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.random.rand(output_size)

def forward_propagation(X):
    hidden_layer_input =    np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    return hidden_layer_output, output

def backpropogation(X, y, hidden_layer_output, output):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output

    output_error = y - output
    output_delta = output_error * sigmoid_derivative(output)

    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * relu_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta)
    bias_output += np.sum(output_delta, axis=0)

    weights_input_hidden += X.T.dot(hidden_layer_delta)
    bias_hidden += np.sum(hidden_layer_delta, axis=0)

def train(X, y, epochs=100000, learning_rate=0.01):
    for epoch in range(epochs):
        hidden_layer_output, output = forward_propagation(X)
        backpropogation(X, y, hidden_layer_output=hidden_layer_output, output=output)

        if epoch % 1000 == 0:
            loss = np.mean((y-output)**2)
            print(f'Epoch {epoch}, Loss: {loss}')

# New Data: Simple binary classification
X_new = np.array([
    [0, 0], [0, 1], [1, 0], [1, 1],
    [0, 2], [2, 0], [2, 2], [1, 2],
    [2, 1], [3, 0], [0, 3], [3, 3]
])
y_new = np.array([
    [0], [1], [1], [0],
    [1], [1], [0], [0],
    [0], [1], [1], [0]
])

# Visualizing the data
import matplotlib.pyplot as plt

plt.scatter(X_new[:, 0], X_new[:, 1], c=y_new[:, 0], cmap='bwr', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('New Binary Classification Data')
plt.show()


train(X_new, y_new)

hidden_layer_output, output = forward_propagation(X_new)
print("Predictions after training:")
print(output)