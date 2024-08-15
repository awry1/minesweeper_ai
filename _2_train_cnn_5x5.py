import os
import time
import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Constants for quick change
SIZE = 10, 10       # X, Y

ITERATIONS = 10     # Iterations in the SMP_Data file
WINDOW_SIZE = 5, 5


def one_hot_encode(board, size_x, size_y):
    one_hot = np.zeros((11, size_x, size_y))  # 10 channels for 0-8 and covered cell (?)
    for idx, val in enumerate(board):
        i = idx // size_y
        j = idx % size_y
        # Map values to corresponding channel
        if val == '_':
            one_hot[10, i, j] = 1  # Out of board cell
        elif val == '?':
            one_hot[9, i, j] = 1  # Undiscovered cell
        else:
            one_hot[int(val), i, j] = 1  # Number of adjacent mines
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
        risk_value = None
        is_risk_line = False

        for line in lines:
            line = line.strip()

            if is_risk_line:
                risk_value = float(line)
                is_risk_line = False
                continue

            if line == 'Risk:':
                is_risk_line = True
                continue

            if not line:
                if input_board and risk_value is not None:
                    numeric_board = self.board_to_numeric(input_board)
                    inputs.append(numeric_board)
                    # Flatten the risk_board to a single value for each board
                    risks.append(np.array([risk_value], dtype=float))
                input_board = []
                risk_value = None
                continue

            input_board.append(line.split())

        # Final append if file does not end with empty line
        if input_board and risk_value is not None:
            numeric_board = self.board_to_numeric(input_board)
            inputs.append(numeric_board)
            risks.append(np.array([risk_value], dtype=float))

        inputs = np.array(inputs, dtype=float)
        risks = np.array(risks, dtype=float)

        print(f"Loaded {len(inputs)} input boards and {len(risks)} risk values")

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
        board, risk_board = self.data[idx]
        board = torch.tensor(board, dtype=torch.float32)
        risk_board = torch.tensor(risk_board, dtype=torch.float32)  # No need to flatten here
        return board, risk_board


class MinesweeperCNN(nn.Module):
    def __init__(self, board_size):
        super(MinesweeperCNN, self).__init__()
        self.size_x, self.size_y = board_size
        self.conv1 = nn.Conv2d(11, 25, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(25, 25, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(25, 64, kernel_size=5, padding=2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(64 * self.size_x * self.size_y, 512)
        self.fc2 = nn.Linear(512, 1)  # Output a single risk value per board

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Single risk value per board
        return x


def train_model(train_loader, board_size, learning_rate, num_epochs, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MinesweeperCNN(board_size).to(device)  # Move model to device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    DIRECTORY = os.path.join('RESULTS_TRAIN', 'CNN')
    os.makedirs(DIRECTORY, exist_ok=True)
    FILENAME = os.path.join(DIRECTORY, f'SMP_TrainResult_s-{SIZE}_'
                            f'e-{num_epochs}_'  # 'hs-{hidden_sizes}_'
                            f'lr-{learning_rate}_wd-{weight_decay}.pth')
    if os.path.exists(FILENAME):
        # Remove the file if it exists
        os.remove(FILENAME)

    print('Using device:', device)
    print('Training model')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for boards, risk_values in train_loader:
            boards = boards.numpy()
            boards_one_hot = np.array([one_hot_encode(b, *board_size) for b in boards])
            boards_one_hot = torch.tensor(boards_one_hot, dtype=torch.float32)

            optimizer.zero_grad()
            boards_one_hot = boards_one_hot.to(device)
            risk_values = risk_values.to(device)
            outputs = model(boards_one_hot)

            # Flatten risk_values for a single risk value per board
            risk_values = risk_values.view(-1, 1)

            loss = criterion(outputs, risk_values)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}\n')

    os.makedirs('MODELS', exist_ok=True)
    FILENAME = os.path.join('MODELS', f'SMP_Model_CNN_{SIZE}.pth')
    torch.save(model.state_dict(), FILENAME)
    print('Model saved:', FILENAME)


if __name__ == '__main__':
    FILENAME = os.path.join('DATASETS', f'SMP_Data_{SIZE}_{ITERATIONS}.txt')

    start_time = time.time()
    print('Loading data:', FILENAME)
    dataset = MinesweeperDataset(FILENAME, WINDOW_SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=5000,  # Adjust batch size as needed
        shuffle=True)
    print(f'Data loaded after: {time.time() - start_time:.2f}s')

    start_time = time.time()
    train_model(
        train_loader,
        board_size=WINDOW_SIZE,
        learning_rate=0.00005,
        num_epochs=300,
        weight_decay=0.000025)
    print(f'Model trained after: {time.time() - start_time:.2f}s')
