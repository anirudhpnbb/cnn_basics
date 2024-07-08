import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(42)

# Benign tumors (0)
benign_size = np.random.normal(5, 1, 50)
benign_smoothness = np.random.normal(3, 0.5, 50)
benign_labels = np.zeros(50)

# Malignant tumors (1)
malignant_size = np.random.normal(10, 1, 50)
malignant_smoothness = np.random.normal(6, 0.5, 50)
malignant_labels = np.ones(50)

# Combine the data
size = np.concatenate([benign_size, malignant_size])
smoothness = np.concatenate([benign_smoothness, malignant_smoothness])
labels = np.concatenate([benign_labels, malignant_labels])

# Convert to PyTorch tensors
X = torch.tensor(np.column_stack((size, smoothness)), dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)


class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    
input_size = 2
model = Perceptron(input_size)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 1000

for epoch in range(epochs):
    model.train()  # Set the model to training mode

    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

model.eval()  # Set the model to evaluation mode

with torch.no_grad():  # No need to calculate gradients for testing
    predictions = model(X)
    predictions = predictions.round()  # Convert probabilities to binary output
    accuracy = (predictions.eq(y).sum() / float(y.shape[0])).item()
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Print predictions
    print(f'Predictions:\n{predictions.view(-1).numpy()}')

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        pred_probs = model(grid)
    Z = pred_probs.reshape(xx.shape).numpy()

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), edgecolors='k', marker='o')
    plt.xlabel('Size')
    plt.ylabel('Smoothness')
    plt.title('Decision Boundary')
    plt.show()

# Plot decision boundary
plot_decision_boundary(model, X.numpy(), y.numpy())
