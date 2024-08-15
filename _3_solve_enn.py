from game import *
from _1_solve_analytical import update_risk_board, choose_least_risky_move
from _2_train_enn import MinesweeperENN
import os
import torch

# Constants for quick change
SIZE = 10, 10       # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0    # Center, Edge, Corner

MOVES_LIMIT = 1     # 0 - no limit
HIDDEN_SIZE = [50, 100, 50, 25]
ITERATIONS = 1000


def load_model(input_size, hidden_size, output_size, model_filename):
    model = MinesweeperENN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_filename))
    model.eval()  # Set the model to evaluation mode
    return model


def take_input_torch(size, num_mines, player_board, game_started, filename, model):
    size_x, size_y = size

    if not game_started:
        row, col = random.randint(0, size_y - 1), random.randint(0, size_x - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
        # Convert player_board list to tensor
        player_board_numerical = convert_board_to_numerical(player_board)
        player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32).view(1, size_x * size_y)

        tensor_risk_board = model(player_board_tensor)

        # Convert PyTorch tensor to NumPy array
        numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

        # Reshape the NumPy array to represent the board structure
        reshaped_board = numpy_risk_board.reshape((size_y, size_x))

        # For debugging purposes
        risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])

        save_torch_results(risk_board, reshaped_board, filename)

        row, col = choose_least_risky_move(reshaped_board)
    
    if not is_input_valid(size, row, col):
        return None, None

    return row, col


def convert_board_to_numerical(board):
    # Define mapping from characters to numerical values
    mapping = {'_': -1.0, '0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, 
               '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, ' ': 9.0}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 10 for cell in row] for row in board]

    return numerical_board


def save_torch_results(true_risk_board, torch_risk_board, filename):
    with open(filename, 'a') as file:
        file.write('True risk factors:\n')

        for row in true_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('Torch risk factors:\n')

        for row in torch_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('\n')


def gameloop_torch(size, default_mines, rand_mines, limits, filename, model, moves_limit):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        row, col = take_input_torch(size, num_mines, player_board, game_started, filename, model)
        
        if (row, col) == last_input or row is None or col is None:
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        if is_mine(game_board, row, col):
            if not game_started:
                return 'L1'
            return 'L'
        
        if game_started:
            moves_limit -= 1
            if moves_limit == 0:
                return 'W'
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

    print('Testing model', end='')
    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'\nProgress: {_}/{iterations}')
        if _ % 5 == 0:
            print('░', end='', flush=True)
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


if __name__ == '__main__':
    MODEL_FILENAME = os.path.join('MODELS', f'Model_ENN_{SIZE}.pth')
    print('Loading model:', MODEL_FILENAME)
    DIRECTORY = os.path.join('RESULTS_TEST', 'ENN')
    os.makedirs(DIRECTORY, exist_ok=True)
    FILENAME = os.path.join(DIRECTORY, f'TestResult_{SIZE}_{DEFAULT_MINES}_{ITERATIONS}.txt')
    simulation(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, MODEL_FILENAME, MOVES_LIMIT, HIDDEN_SIZE, SEED, ITERATIONS)
