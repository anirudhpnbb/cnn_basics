import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define transformations for the training set
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Load the training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the Neural Network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer (784 -> 128)
        self.fc2 = nn.Linear(128, 64)       # Fully connected layer (128 -> 64)
        self.fc3 = nn.Linear(64, 10)        # Fully connected layer (64 -> 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input (batch_size, 1, 28, 28) -> (batch_size, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model
model = SimpleNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training the model
epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}')

print("Training complete.")

# Evaluating the model
model.eval()  # Set the model to evaluation mode

correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10,000 test images: {accuracy:.2f}%')

# Print predictions for a few samples
print("\nSample predictions:")
with torch.no_grad():
    images, labels = next(iter(testloader))
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    
    # Print the true labels and predicted labels for the first 10 samples
    print("True labels:     ", labels[:10].numpy())
    print("Predicted labels:", predicted[:10].numpy())
