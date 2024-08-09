import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Constants for quick change
SIZE = 10, 10  # X, Y
ITERATIONS = 400  # Iterations in the data file


def one_hot_encode(board, size_x, size_y):
    one_hot = np.zeros((10, size_x, size_y))  # 10 channels for 0-8 and covered cell (?)
    for idx, val in enumerate(board):
        i = idx // size_y
        j = idx % size_y
        one_hot[int(val), i, j] = 1
    return one_hot


class MinesweeperDataset(Dataset):
    def __init__(self, file_path, board_size):
        self.size_x, self.size_y = board_size
        self.data = self.load_data(file_path)
        print(f"Loaded {len(self.data)} samples")

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        inputs = []
        risks = []

        input_board = []
        risk_board = []
        is_risk_line = False

        for line in lines:
            line = line.strip()

            if is_risk_line:
                risk_board.append(list(map(float, line.split())))
                if len(risk_board) == self.size_x:
                    risks.append(np.array(risk_board, dtype=float))
                    risk_board = []
                    is_risk_line = False
                continue

            if line == 'Risk factors:':
                is_risk_line = True
                continue

            if not line:
                if input_board and not risk_board:
                    inputs.append(self.board_to_numeric(input_board))
                    input_board = []
                continue

            input_board.append(line.split())

        # Final append if file does not end with empty line
        if input_board and not risk_board:
            inputs.append(self.board_to_numeric(input_board))

        inputs = np.array(inputs, dtype=float)
        risks = np.array(risks, dtype=float)

        print(f"Loaded {len(inputs)} input boards and {len(risks)} risk maps")

        return list(zip(inputs, risks))

    def board_to_numeric(self, board):
        mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '?': 9.0, '_': 9.0, ' ': 9.0}
        numeric_board = np.zeros((self.size_x, self.size_y), dtype=float)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.size_x and j < self.size_y:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        board, risk_board = self.data[idx]
        board = torch.tensor(board, dtype=torch.float32)
        risk_board = torch.tensor(risk_board, dtype=torch.float32)
        return board, risk_board


class MinesweeperCNN(nn.Module):
    def __init__(self, board_size):
        super(MinesweeperCNN, self).__init__()
        self.size_x, self.size_y = board_size
        self.conv1 = nn.Conv2d(10, 25, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(25, 64, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * self.size_x * self.size_y, 512)
        self.fc2 = nn.Linear(512, self.size_x * self.size_y)  # Output a risk value per tile

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Risk value per tile
        x = x.view(-1, self.size_x, self.size_y)
        return x


def train_model(train_loader, board_size, learning_rate, num_epochs, weight_decay):
    model = MinesweeperCNN(board_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    os.makedirs('RESULTS_TRAIN', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TRAIN', f'TrainResult_s-{board_size}_'
                                             f'e-{num_epochs}_lr-{learning_rate}_wd-{weight_decay}.pth')
    if os.path.exists(FILENAME):
        os.remove(FILENAME)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for boards, risk_values in train_loader:
            boards = boards.numpy()
            boards_one_hot = np.array([one_hot_encode(b, *board_size) for b in boards])
            boards_one_hot = torch.tensor(boards_one_hot, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(boards_one_hot)

            loss = criterion(outputs, risk_values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}\n')

    os.makedirs('MODELS', exist_ok=True)
    FILENAME = os.path.join('MODELS', f'Model_{board_size}_cnn.pth')

    torch.save(model.state_dict(), FILENAME)


if __name__ == '__main__':
    FILENAME = os.path.join('DATA', f'Data_{SIZE}_{ITERATIONS}.txt')

    print('Using data file:', FILENAME)
    print('Loading Data...')
    dataset = MinesweeperDataset(FILENAME, SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=8192,  # Adjust batch size as needed
        shuffle=True)
    print('Data Loaded')

    train_model(
        train_loader,
        board_size=SIZE,
        learning_rate=0.00005,
        num_epochs=100,
        weight_decay=0.000025)