import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Constants for quick change
SIZE = 10

ITERATIONS_ANALYTICAL = 1000

MODEL_NAME = 'Model'
TRAIN_NAME = 'Data'
RESULTS_NAME = 'TrainResult'

MODEL_DIRECTORY = 'MODELS'
TRAIN_DIRECTORY = 'DATA'
RESULTS_DIRECTORY = 'RESULTS_TRAIN'


# Creating custom dataset for boards and moves
class MinesweeperDataset(Dataset):
    def __init__(self, file_path, board_size):
        self.board_size = board_size
        self.data = self.load_data(file_path)


    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        inputs = []
        risks = []
        mines = []

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

            if line.startswith('Number of mines:'):
                loading_mines = True
                continue

            if line.startswith('Risk factors:'):
                loading_risk = True
                continue

            if loading_risk:
                risk_board.append([float(val) for val in line.split()])
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
        mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, '?': 9.0}
        numeric_board = np.zeros((self.board_size, self.board_size), dtype=float)

        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if i < self.board_size and j < self.board_size:
                    numeric_board[i][j] = mapping.get(cell, 0)

        return numeric_board.flatten()


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
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            self.hidden_layers.extend([nn.Linear(hidden_size[i], hidden_size[i+1])])
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        self.dropout = nn.Dropout(p=0.3)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size[i]) for i in range(len(hidden_size))])

    def forward(self, boards):
        x = boards
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = torch.relu(bn(layer(x)))
            x = self.dropout(x)

        risk_board = torch.sigmoid(self.output_layer(x))

        return risk_board


# Step 3 & 4: Define Loss Function and Optimizer and create Train Iteration Loop
def train_model(train_loader, input_size, output_size, hidden_sizes, learning_rate, num_epochs, weight_decay):
    model = MinesweeperMLP(input_size, hidden_sizes, output_size)  # Move model to device
    criterion = nn.MSELoss()  # You can try nn.L1Loss() or a custom loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)
    FILENAME = os.path.join(RESULTS_DIRECTORY, f'{RESULTS_NAME}_s-{SIZE}_e-{num_epochs}_hs-{hidden_sizes}_'
                                               f'lr-{learning_rate}_wd-{weight_decay}.pth')
    if os.path.exists(FILENAME):
        # Remove the file if it exists
        os.remove(FILENAME)

    # Run for a fixed number of epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # num_mines
        for boards, risk_boards in train_loader:
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, risk_boards)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}')
        with open(FILENAME, 'a') as file:
            file.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}\n')

    # Run till good
    """ running_loss = 1.0
    while running_loss > 0.01:
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss}") """
    
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    FILENAME = os.path.join(MODEL_DIRECTORY, f'{MODEL_NAME}_{SIZE}.pth')

    torch.save(model.state_dict(), FILENAME)

if __name__ == '__main__':
    FILENAME = os.path.join(TRAIN_DIRECTORY, f'{TRAIN_NAME}_{SIZE}_{ITERATIONS_ANALYTICAL}.txt')

    print('Loading Data...')
    dataset = MinesweeperDataset(FILENAME, SIZE)
    train_loader = DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True)
    print('Data Loaded')

    # batch_size=8192,
    # input_size=(board_size * board_size),
    # output_size=(board_size * board_size),
    # hidden_sizes=[1000, 1000],
    # learning_rate=0.00005,
    # num_epochs=100,
    # weight_decay=0.000025)

    train_model(
        train_loader,
        input_size=(SIZE * SIZE),
        output_size=(SIZE * SIZE),
        hidden_sizes=[64],
        learning_rate=0.00005,
        num_epochs=300,
        weight_decay=0.000025)
