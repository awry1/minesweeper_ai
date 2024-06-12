import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re
from sklearn.preprocessing import StandardScaler

class MinesweeperDataset(Dataset):
    def __init__(self, file_path, board_size):
        self.board_size = board_size
        self.scaler = StandardScaler()
        self.data = self.load_data(file_path)
        self.fit_scaler()

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

    def fit_scaler(self):
        all_boards = np.array([board for board, _ in self.data])
        self.scaler.fit(all_boards)

    def board_to_numeric(self, board):
        mapping = {'?': 0, 'X': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10}
        numeric_board = np.zeros((self.board_size, self.board_size), dtype=int)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.board_size and j < self.board_size:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()

    def is_valid_move(self, line):
        pattern = r'^\d+\s+\d+$'
        return bool(re.match(pattern, line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, move = self.data[idx]
        board = self.scaler.transform(board.reshape(1, -1))  # Normalize input
        board = torch.tensor(board, dtype=torch.float32)
        move = torch.tensor(move, dtype=torch.float32).view(1, -1)  # Reshape target tensor
        return board, move


class MinesweeperNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MinesweeperNN, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1])])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

def train_model(train_loader, input_size, hidden_sizes, output_size, learning_rate=0.0001, num_epochs=300, weight_decay=0.001):
    model = MinesweeperNN(input_size, hidden_sizes, output_size).to(device)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'minesweeper_nn.pth')

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    file_path = 'trainingData.txt'
    board_size = 10
    dataset = MinesweeperDataset(file_path, board_size)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    input_size = board_size * board_size
    hidden_sizes = [200]  # Define more dense layers
    output_size = 2

    train_model(train_loader, input_size, hidden_sizes, output_size)
