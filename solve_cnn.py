from game import *
from solve_nn import save_torch_results
from solve_analytical import update_risk_board, choose_least_risky_move, find_undiscovered_fields
from train_cnn import MinesweeperCNN, one_hot_encode
import os
import torch
import numpy as np

# Constants for quick change
SIZE = 10, 10       # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0    # Center, Edge, Corner

MOVES_LIMIT = 0     # 0 - no limit
ITERATIONS = 1000


def load_model(input_size, model_filename):
    model = MinesweeperCNN(input_size)  # Initialize the model class
    model.load_state_dict(torch.load(model_filename, weights_only=True))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode
    return model


def take_input_torch_10x10(size, num_mines, player_board, game_started, filename, model, window_size):
    size_x, size_y = size

    if not game_started:
        row, col = random.randint(0, size_x - 1), random.randint(0, size_y - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
        # What is this even
        undiscovered = find_undiscovered_fields(player_board)
        torch_risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        for field, _ in undiscovered:
            row, col = field
            #board = create_window(player_board, field, window_size)
            board = board_to_string(player_board)
            board = one_hot_encode(board, window_size[0], window_size[1])
            window_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)  # Shape: [batch_size, channels, height, width]
            torch_risk = model(window_tensor)
            torch_risk_board[row][col] = torch_risk.view(size_x, size_y)[row, col].item()

        """ Copied from solve_nn.py
        # Convert player_board list to tensor
        player_board_numerical = convert_board_to_numerical(player_board)
        player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32).view(1, size_x * size_y)

        tensor_risk_board = model(player_board_tensor)

        # Convert PyTorch tensor to NumPy array
        numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

        # Reshape the NumPy array to represent the board structure
        reshaped_board = numpy_risk_board.reshape((size_y, size_x))
        """

        # For debugging purposes
        risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])

        save_torch_results(risk_board, torch_risk_board, filename)

        row, col = choose_least_risky_move(torch_risk_board)

    if not is_input_valid(size, row, col):
        return None, None

    return row, col


def board_to_string(board, unknown_value=9.0, artificial_value=9.0):
    # Flatten the 2D board list into a 1D list of floats
    flattened_board = [
        unknown_value if cell == '?' or cell == ' ' else
        artificial_value if cell == '_' else
        float(cell)  # Convert numeric characters to float
        for row in board for cell in row
    ]
    # Convert the list into a numpy ndarray with shape (25,)
    board_ndarray = np.array(flattened_board, dtype=float)
    return board_ndarray


def gameloop_torch_10x10(size, default_mines, rand_mines, limits, filename, model, window_size, moves_limit):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        row, col = take_input_torch_10x10(size, num_mines, player_board, game_started, filename, model, window_size)

        if (row, col) == last_input or row is None or col is None:
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        if is_mine(game_board, row, col):
            if not game_started:
                return 'L1'
            return 'L'
        else:
            if game_started:
                moves_limit -= 1
                if moves_limit == 0:
                    return 'W'
            else:
                game_started = True
            reveal_squares(game_board, player_board, row, col)

            if is_game_finished(game_board, player_board):
                return 'W'


def simulation_10x10(size, default_mines, rand_mines, limits, filename, model_filename, window_size, moves_limit, seed, iterations):
    model = load_model(window_size, model_filename)

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    print('Testing model', end='')
    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'\nProgress: {_}/{iterations}')
        if _ % 5 == 0:
            print('░', end='', flush=True)
        result = gameloop_torch_10x10(size, default_mines, rand_mines, limits, filename, model, window_size, moves_limit)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1
    print(f'\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}')

    with open(filename, 'r') as file:
        existing_content = file.read()
    with open(filename, 'w') as file:
        file.write(f'Wins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}\n')
        file.write('\n')
        file.write(existing_content)


if __name__ == '__main__':
    MODEL_FILENAME = os.path.join('MODELS', f'Model_CNN_{SIZE}.pth')
    print('Using model file:', MODEL_FILENAME)
    os.makedirs('RESULTS_TEST', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TEST', f'TestResult_{SIZE}_{ITERATIONS}_{DEFAULT_MINES}.txt')
    simulation_10x10(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, MODEL_FILENAME, WINDOW_SIZE, MOVES_LIMIT, SEED, ITERATIONS)
