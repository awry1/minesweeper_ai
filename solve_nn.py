from solve_analytical import *
from train_nn import MinesweeperMLP
import os
import torch

# Constants for quick change
SIZE = 5, 5     # X, Y
DEFAULT_MINES = 4
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0    # Center, Edge, Corner

MOVES_LIMIT = 1 # 0 - no limit
HIDDEN_SIZE = [64]
ITERATIONS = 1000


def load_model(input_size, hidden_size, output_size, model_filename):
    model = MinesweeperMLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set the model to evaluation mode
    return model


def take_input_torch(size, num_mines, player_board, game_started, model, filename):
    size_x, size_y = size
    row, col = None, None

    if not game_started:
        row, col = random.randint(0, size_y - 1), random.randint(0, size_x - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
        # Convert player_board list to tensor
        player_board_numerical = convert_board_to_numerical(player_board)
        player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32).view(1, size_x * size_y)

        # For debugging purposes
        risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])

        tensor_risk_board = model(player_board_tensor)

        # Convert PyTorch tensor to NumPy array
        numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

        # Reshape the NumPy array to represent the board structure
        reshaped_board = numpy_risk_board.reshape((size_y, size_x))

        save_torch_results(risk_board, num_mines, reshaped_board, filename)

        row, col = choose_least_risky_move(reshaped_board)
    
    if not is_input_valid(size, row, col):
        return None, None

    # print('\nAI chose:', col, row)
    return row, col


def convert_board_to_numerical(board):
    # Define mapping from characters to numerical values
    mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, ' ': 9.0}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 0 for cell in row] for row in board]

    return numerical_board


def save_torch_results(true_risk_board, torch_num_mines, torch_risk_board, filename):
    with open(filename, 'a') as file:
        file.write('True risk factors:\n')

        for row in true_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('Torch risk factors:\n')

        for row in torch_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('Torch number of mines:\n')
        file.write(str(torch_num_mines) + '\n')
        file.write('\n')


def gameloop_torch(size, default_mines, rand_mines, limits, filename, model, moves_limit):
    size_x, size_y = size
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        #risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        #risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])
        #true_row, true_col = choose_least_risky_move(risk_board)

        row, col = take_input_torch(size, num_mines, player_board, game_started, model, filename)
        
        if (row, col) == last_input or row is None or col is None:
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        #  true_row != row or true_col != col
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


def simulation(size, default_mines, rand_mines, limits, filename, model_filename, moves_limit, hidden_size, seed, iterations):
    size_x, size_y = size
    model = load_model((size_x * size_y), hidden_size, (size_x * size_y), model_filename)

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'Progress: {_}/{iterations}')
        result = gameloop_torch(size, default_mines, rand_mines, limits, filename, model, moves_limit)
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
    quit()


if __name__ == '__main__':
    MODEL_FILENAME = os.path.join('MODELS', f'Model_{SIZE}.pth')
    print('Using model file:', MODEL_FILENAME)
    os.makedirs('RESULTS_TEST', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TEST', f'TestResult_{SIZE}_{ITERATIONS}.txt')
    simulation(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, MODEL_FILENAME, MOVES_LIMIT, HIDDEN_SIZE, SEED, ITERATIONS)
