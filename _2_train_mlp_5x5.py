from _2_train_mlp import MinesweeperMLP
import os
import time
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Constants for quick change
SIZE = 10, 10       # X, Y

ITERATIONS = 1000   # Iterations in the SMP_Data file
WINDOW_SIZE = 5, 5
ENCHANCE = False


# Creating custom dataset for boards and moves
class MinesweeperDataset5x5(Dataset):
    def __init__(self, file_path, window_size):
        self.size_x, self.size_y = window_size
        self.data = self.load_data_5x5(file_path)

    def load_data_5x5(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        inputs = []
        risks = []

        input_board = []
        risk_board = []

        loading_risk = False
        for line in lines:
            line = line.strip()

            if len(line) == 0:
                loading_risk = False
                if input_board and risk_board:
                    inputs.append(self.board_to_numeric(input_board))
                    risks.append(np.array(risk_board, dtype=float).flatten())
                input_board = []
                risk_board = []
                continue

            if line.startswith('Risk:'):
                loading_risk = True
                continue

            if loading_risk:
                risk_board.append(line.split())
            else:
                input_board.append(line.split())

        # Final append if file does not end with empty line
        if input_board and risk_board:
            inputs.append(self.board_to_numeric(input_board))
            risks.append(np.array(risk_board, dtype=float).flatten())

        inputs = np.array(inputs, dtype=float)
        risks = np.array(risks, dtype=float)

        # Normalize input and output data
        inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
        risks = (risks - np.min(risks)) / (np.max(risks) - np.min(risks))

        return list(zip(inputs, risks))

    def board_to_numeric(self, board):
        mapping = {'_': -1.0, '0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, 
                   '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '?': 9.0}
        numeric_board = np.zeros((self.size_x, self.size_y), dtype=float)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.size_x and j < self.size_y:
                    numeric_board[i][j] = mapping.get(cell, 10)

        return numeric_board.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, move = self.data[idx]
        window = torch.tensor(window, dtype=torch.float32)
        risk = torch.tensor(move, dtype=torch.float32)
        return window, risk


# Step 2: Define the model
# Imported from train_nn.py


# Step 3 & 4: Define loss function and optimizer and create training loop
def train_model(train_loader, input_size, output_size, hidden_sizes, learning_rate, num_epochs, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinesweeperMLP(input_size, hidden_sizes, output_size).to(device)  # Move model to device
    criterion = nn.MSELoss()  # You can try nn.L1Loss() or a custom loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    DIRECTORY = os.path.join('RESULTS_TRAIN', 'MLP')
    os.makedirs(DIRECTORY, exist_ok=True)
    FILENAME = os.path.join(DIRECTORY, f'SMP_TrainResult_s-{SIZE}_'
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
        # num_mines
        for window, risk in train_loader:
            optimizer.zero_grad()
            window = window.to(device)
            risk = risk.to(device)
            outputs = model(window)
            loss = criterion(outputs, risk)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}\n')
    
    os.makedirs('MODELS', exist_ok=True)
    FILENAME = os.path.join('MODELS', f'SMP_Model_MLP_{SIZE}.pth')
    torch.save(model.state_dict(), FILENAME)
    print('Model saved:', FILENAME)


def enchance_model(train_loader, input_size, output_size, hidden_sizes, learning_rate, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinesweeperMLP(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.MSELoss()  # You can try nn.L1Loss() or a custom loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Load the model if it exists
    FILENAME = os.path.join('MODELS', f'SMP_Model_MLP_{SIZE}.pth')
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
        for window, risk in train_loader:
            optimizer.zero_grad()
            window = window.to(device)
            risk = risk.to(device)
            outputs = model(window)
            loss = criterion(outputs, risk)
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
    FILENAME = os.path.join('DATASETS', f'SMP_Data_{SIZE}_{ITERATIONS}.txt')
    size_x, size_y = WINDOW_SIZE

    start_time = time.time()
    print('Loading data:', FILENAME)
    dataset = MinesweeperDataset5x5(FILENAME, WINDOW_SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=5000,
        shuffle=True)
    print(f'Data loaded after: {time.time() - start_time:.2f}s')

    start_time = time.time()
    if ENCHANCE:
        enchance_model(
            train_loader,
            input_size=(size_x * size_y),
            output_size=(1),
            hidden_sizes=[50, 100, 50, 25],
            learning_rate=0.00005,
            weight_decay=0.000025)
    else:
        train_model(
            train_loader,
            input_size=(size_x * size_y),
            output_size=(1),
            hidden_sizes=[50, 100, 50, 25],
            learning_rate=0.00005,
            num_epochs=300,
            weight_decay=0.000025)
    print(f'Model trained after: {time.time() - start_time:.2f}s')