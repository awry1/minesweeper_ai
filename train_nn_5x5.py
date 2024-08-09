from train_nn import MinesweeperMLP
import os
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# Constants for quick change
SIZE = 10, 10  # X, Y

ITERATIONS = 1000  # Iterations in the data file
WINDOW_SIZE = 5, 5


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
        mapping = {'_': -1.0, '0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0,
                   '?': 9.0}
        numeric_board = np.zeros((self.size_x, self.size_y), dtype=float)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.size_x and j < self.size_y:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window, move = self.data[idx]
        window = torch.tensor(window, dtype=torch.float32)
        risk = torch.tensor(move, dtype=torch.float32)
        return window, risk


def train_model_5x5(train_loader, input_size, output_size, hidden_sizes, learning_rate, num_epochs, weight_decay):
    model = MinesweeperMLP(input_size, hidden_sizes, output_size)  # Move model to device
    criterion = nn.MSELoss()  # You can try nn.L1Loss() or a custom loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    os.makedirs('RESULTS_TRAIN', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TRAIN', f'SMP_TrainResult_s-{SIZE}_'
                                             f'e-{num_epochs}_hs-{hidden_sizes}_'
                                             f'lr-{learning_rate}_wd-{weight_decay}.pth')
    if os.path.exists(FILENAME):
        # Remove the file if it exists
        os.remove(FILENAME)

    # Run for a fixed number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # num_mines
        for window, risk in train_loader:
            optimizer.zero_grad()
            outputs = model(window)
            loss = criterion(outputs, risk)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss}\n')
    
    """ # Run till good
    running_loss = 1.0
    epoch = 0
    while running_loss > 0.01:
        model.train()
        running_loss = 0.0
        for window, risk in train_loader:
            optimizer.zero_grad()
            outputs = model(window)
            loss = criterion(outputs, risk)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch}/till good, Loss: {running_loss}')
        epoch += 1 """

    os.makedirs('MODELS', exist_ok=True)
    FILENAME = os.path.join('MODELS', f'SMP_Model_{SIZE}_nn.pth')

    torch.save(model.state_dict(), FILENAME)


if __name__ == '__main__':
    FILENAME = os.path.join('DATA', f'SMP_Data_{SIZE}_{ITERATIONS}.txt')

    print('Using data file:', FILENAME)
    print('Loading Data...')
    dataset = MinesweeperDataset5x5(FILENAME, WINDOW_SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=32768, # 65536 131072
        shuffle=True)
    print('Data Loaded')

    size_x, size_y = WINDOW_SIZE
    train_model_5x5(
        train_loader,
        input_size=(size_x * size_y),
        output_size=(1),
        hidden_sizes=[50, 100, 50, 25],
        learning_rate=0.00003,
        num_epochs=200,
        weight_decay=0.000025)
