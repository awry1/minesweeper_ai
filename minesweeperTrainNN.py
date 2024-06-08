import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re

class MinesweeperDataset(Dataset):
    def __init__(self, file_path, board_size):
        self.board_size = board_size
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        boards = []
        moves = []
        board = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            if not self.is_valid_move(line):
                board.append(line.split())
            else:
                moves.append([int(x) for x in line.split()])
                boards.append(self.board_to_numeric(board))
                board = []

        moves = np.array(moves)
        return list(zip(boards, moves))

    def board_to_numeric(self, board):
        mapping = {'?': 0, 'X': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10}
        numeric_board = np.zeros((self.board_size, self.board_size), dtype=int)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.board_size and j < self.board_size:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()

    def is_valid_move(self, line):
        # Regular expression pattern to match a tuple of integers
        pattern = r'^\d+\s+\d+$'
        return bool(re.match(pattern, line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move = self.data[idx]
        board = torch.tensor(board, dtype=torch.float32)
        move = torch.tensor(move, dtype=torch.long)
        return board, move

# Step 2: Define the Model
class MinesweeperNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinesweeperNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Step 3 & 4: Define Loss Function and Optimizer and create Train Iteration Loop
def train_model(train_loader, input_size, hidden_size, output_size, learning_rate=0.0001, num_epochs=300, weight_decay=0.001):
    model = MinesweeperNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for boards, moves in train_loader:
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, moves.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'minesweeper_nn.pth')

if __name__ == '__main__':
    file_path = 'trainingData.txt'
    board_size = 10  # Change this to the size of your boards
    dataset = MinesweeperDataset(file_path, board_size)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    input_size = board_size * board_size  # Example for a 10x10 board
    hidden_size = 500
    output_size = 2  # Move coordinates (row, col)

    train_model(train_loader, input_size, hidden_size, output_size)