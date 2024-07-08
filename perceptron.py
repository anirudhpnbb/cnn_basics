
import torch
import torch.nn as nn
import torch.optim as optim

class Perceptron(nn.Module):
    def __init__(self, input_size):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    
input_size = 2
model = Perceptron(input_size=input_size)

X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs=1000

for epoch in range(epochs):
    model.train()
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item(): .4f}")

print("Training completed")

model.eval()

with torch.no_grad():
    predictions = model(X)
    predictions = predictions.round()
    print(f"Predictions: \n{predictions}")
