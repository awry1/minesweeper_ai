import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import re

# Creating custom dataset for boards and moves
class MinesweeperDataset(Dataset):
    def __init__(self, file_path, board_size):
        self.board_size = board_size
        self.data = self.load_data(file_path)


    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        in_boards = []
        out_boards = []
        input_taken = True
        board = []
        risk_board = np.empty((0, 0), dtype=float)
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                in_boards.append(self.board_to_numeric(board))
                out_boards.append(risk_board.flatten())
                input_taken = True
                board = []
                risk_board = np.zeros((0, 0), dtype=float)  # Reset risk_board
                continue
            if line.startswith('Move applied:'):
                input_taken = False
                continue

            if input_taken:
                board.append(line.split())
            else:
                risk_board = np.append(risk_board, line.split())

        risk_boards = np.array(out_boards, dtype=float)
        in_boards = np.array(in_boards, dtype=float)

        return list(zip(in_boards, risk_boards))


    def board_to_numeric(self, board):
        mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '?': 9.0}
        numeric_board = np.zeros((self.board_size, self.board_size), dtype=float)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.board_size and j < self.board_size:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()


    def is_valid_move(self, line):
        # Checks if pattern matches move coordinates
        pattern = 'Moves risk factor:'
        return bool(re.match(pattern, line.strip()))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        board, move = self.data[idx]
        board = torch.tensor(board, dtype=torch.float32)
        risk_board = torch.tensor(move, dtype=torch.float32)
        return board, risk_board


# Step 2: Define the Model
class MinesweeperMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MinesweeperMLP, self).__init__()
        self.hidden_layer = nn.ModuleList([nn.Linear(input_size, hidden_size[0])])
        for i in range(len(hidden_size) - 1):
            self.hidden_layer.extend([nn.Linear(hidden_size[i], hidden_size[i+1])])
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        for layer in self.hidden_layer:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x


# Step 3 & 4: Define Loss Function and Optimizer and create Train Iteration Loop
def train_model(train_loader, input_size, hidden_sizes, output_size, learning_rate=0.001, num_epochs=150, weight_decay=0.001):
    model = MinesweeperMLP(input_size, hidden_sizes, output_size)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for boards, risk_boards in train_loader:
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, risk_boards)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}")

    torch.save(model.state_dict(), 'minesweeper_nn.pth')

if __name__ == '__main__':
    file_path = 'trainingData.txt'
    board_size = 10
    dataset = MinesweeperDataset(file_path, board_size)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    train_model(train_loader, input_size=(board_size * board_size), hidden_sizes=[800,600,400,200], output_size=(board_size * board_size))
