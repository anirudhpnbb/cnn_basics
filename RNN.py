import torch
import torch.nn as nn
import torch.optim as optim

# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Hyperparameters
input_size = 1
hidden_size = 128
output_size = 1
learning_rate = 0.005
n_iters = 1000
print_every = 100

# Initialize the model, loss function, and optimizer
rnn = RNN(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(rnn.parameters(), lr=learning_rate)

# Generate some sample sequential data
def generate_data(seq_length, num_sequences):
    X = torch.rand(num_sequences, seq_length, input_size)
    y = torch.sum(X, dim=1)
    return X, y

seq_length = 10
num_sequences = 100
X, y = generate_data(seq_length, num_sequences)

# Training loop
for iter in range(1, n_iters + 1):
    for i in range(num_sequences):
        rnn.zero_grad()
        hidden = rnn.initHidden()
        
        loss = 0
        for j in range(seq_length):
            output, hidden = rnn(X[i][j].view(1, -1), hidden)
            loss += criterion(output, y[i].view(1, -1))
        
        loss.backward()
        optimizer.step()
    
    if iter % print_every == 0:
        print(f"Iter {iter} / {n_iters}, Loss: {loss.item()}")

# Testing with a new sequence
test_seq, test_target = generate_data(seq_length, 1)
hidden = rnn.initHidden()
for i in range(seq_length):
    output, hidden = rnn(test_seq[0][i].view(1, -1), hidden)

print("Predicted:", output.item())
print("Actual:", test_target.item())
