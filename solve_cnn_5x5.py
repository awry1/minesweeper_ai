from game import *
from solve_nn import load_model, save_torch_results, convert_board_to_numerical
from solve_analytical import update_risk_board, choose_least_risky_move, find_undiscovered_fields
from solve_analytical_5x5 import create_window
from CNN import MinesweeperCNN, one_hot_encode
from CNN_test import board_to_string1
import os
import torch

# Constants for quick change
SIZE = 10, 10  # X, Y
DEFAULT_MINES = 4
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0  # Center, Edge, Corner

MOVES_LIMIT = 0  # 0 - no limit
HIDDEN_SIZE = [50, 100, 50, 25]
ITERATIONS = 1000
WINDOW_SIZE = 5, 5


def take_input_torch_5x5(size, num_mines, player_board, game_started, filename, model, window_size):
    size_x, size_y = size
    window_x, window_y = window_size

    if not game_started:
        row, col = random.randint(0, size_x - 1), random.randint(0, size_y - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
        undiscovered = find_undiscovered_fields(player_board)
        torch_risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        for field, _ in undiscovered:
            row, col = field
            window = create_window(player_board, field, window_size)
            #window_numerical = convert_board_to_numerical(window)
            window = board_to_string1(window)
            window_numerical = one_hot_encode(window, window_x, window_y)
            window_tensor = torch.tensor(window_numerical, dtype=torch.float32).unsqueeze(0)  # Shape: [batch_size, channels, height, width]
            torch_risk = model(window_tensor)
            torch_risk_board[row][col] = torch_risk.item()

        # For debugging purposes
        risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])

        save_torch_results(risk_board, torch_risk_board, filename)

        row, col = choose_least_risky_move(torch_risk_board)

    if not is_input_valid(size, row, col):
        return None, None

    # print('\nAI chose:', col, row)
    return row, col




def gameloop_torch_5x5(size, default_mines, rand_mines, limits, filename, model, window_size, moves_limit):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        row, col = take_input_torch_5x5(size, num_mines, player_board, game_started, filename, model, window_size)

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



def simulation_5x5(size, default_mines, rand_mines, limits, filename, model_filename, window_size, moves_limit, hidden_size, seed, iterations):
    SIZE_t = (5, 5)
    model = MinesweeperCNN(SIZE_t)  # Initialize the model class
    model.load_state_dict(torch.load(model_filename))  # Load the trained weights
    model.eval()  # Set the model to evaluation mode

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'\nProgress: {_}/{iterations}')
        if _ % 5 == 0:
            print('░', end='', flush=True)
        result = gameloop_torch_5x5(size, default_mines, rand_mines, limits, filename, model, window_size, moves_limit)
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
    MODEL_FILENAME = os.path.join('MODELS', f'Model_(5, 5).pth')
    print('Using model file:', MODEL_FILENAME)
    os.makedirs('RESULTS_TEST', exist_ok=True)
    FILENAME = os.path.join('RESULTS_TEST', f'SMP_TestResult_{SIZE}_{ITERATIONS}.txt')
    simulation_5x5(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, MODEL_FILENAME, WINDOW_SIZE, MOVES_LIMIT, HIDDEN_SIZE, SEED, ITERATIONS)
