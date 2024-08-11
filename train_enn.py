from train_nn import MinesweeperDataset
import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Constants for quick change
SIZE = 10, 10       # X, Y

ITERATIONS = 1000    # Iterations in the Data file
ENCHANCE = True


# Creating custom dataset for boards and moves
# Imported from train_nn.py


# Step 2: Define the model
class MinesweeperENN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MinesweeperENN, self).__init__()
        
        # Encoder (as in the autoencoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[3]),
        )
        
        # Prediction layer (instead of a decoder)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Linear(hidden_sizes[2], output_size),
            nn.Sigmoid()  # Ensure outputs are between 0 and 1
        )

    def forward(self, boards):
        encoded = self.encoder(boards)
        risk_board = self.predictor(encoded)
        return risk_board


# Step 3 & 4: Define loss function and optimizer and create training loop
def train_model(train_loader, input_size, output_size, hidden_sizes, learning_rate, num_epochs, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinesweeperENN(input_size, hidden_sizes, output_size).to(device)  # Move model to device
    criterion = nn.MSELoss()  # Loss function remains the same
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    os.makedirs('RESULTS_TRAIN', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TRAIN', f'TrainResult_s-{SIZE}_'
                            f'e-{num_epochs}_hs-{hidden_sizes}_'
                            f'lr-{learning_rate}_wd-{weight_decay}.pth')
    if os.path.exists(FILENAME):
        # Remove the file if it exists
        os.remove(FILENAME)

    # Run for a fixed number of epochs
    print('Using device:', device)
    print('Training model')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for board, risk_board in train_loader:
            optimizer.zero_grad()
            board = board.to(device)
            risk_board = risk_board.to(device)
            outputs = model(board)
            loss = criterion(outputs, risk_board)  # Compare the output with the risk_board
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}\n')

    os.makedirs('MODELS', exist_ok=True)
    FILENAME = os.path.join('MODELS', f'Model_ENN_{SIZE}.pth')
    torch.save(model.state_dict(), FILENAME)
    print('Model saved:', FILENAME)


def enchance_model(train_loader, input_size, output_size, hidden_sizes, learning_rate, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinesweeperENN(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.MSELoss()  # Loss function remains the same
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Load the model if it exists
    FILENAME = os.path.join('MODELS', f'Model_ENN_{SIZE}.pth')
    if not os.path.exists(FILENAME):
        print('Unable to find model:', FILENAME)
        quit()
    model.load_state_dict(torch.load(FILENAME, map_location=device))
    print('Loading model:', FILENAME)

    # Run till good with a periodic save
    print('Using device:', device)
    print('Training model')
    running_loss = 1.0
    epoch = 1
    while running_loss > 0.01:
        model.train()
        running_loss = 0.0
        for board, risk_board in train_loader:
            optimizer.zero_grad()
            board = board.to(device)
            risk_board = risk_board.to(device)
            outputs = model(board)
            loss = criterion(outputs, risk_board)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch}, Loss: {running_loss}')
        if epoch % 100 == 0:
            torch.save(model.state_dict(), FILENAME)
            print('Model saved:', FILENAME)
        epoch += 1

    torch.save(model.state_dict(), FILENAME)
    print('Model saved:', FILENAME)

if __name__ == '__main__':
    FILENAME = os.path.join('DATA', f'Data_{SIZE}_{ITERATIONS}.txt')
    size_x, size_y = SIZE

    start_time = time.time()
    print('Loading data:', FILENAME)
    dataset = MinesweeperDataset(FILENAME, SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=100000,
        shuffle=True)
    print(f'Data loaded after: {time.time() - start_time:.2f}s')

    start_time = time.time()
    if ENCHANCE:
        enchance_model(
            train_loader,
            input_size=(size_x * size_y),
            output_size=(size_x * size_y),
            hidden_sizes=[50, 100, 50, 25],
            learning_rate=0.00005,
            weight_decay=0.000025)
    else:
        train_model(
            train_loader,
            input_size=(size_x * size_y),
            output_size=(size_x * size_y),
            hidden_sizes=[50, 100, 50, 25],
            learning_rate=0.00005,
            num_epochs=300,
            weight_decay=0.000025)
    print(f'Model trained after: {time.time() - start_time:.2f}s')
