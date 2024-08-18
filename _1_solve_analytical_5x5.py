from game import *
from _1_solve_analytical import solve_analytical, choose_least_risky_move, find_undiscovered_fields, update_risk_board
import os
import time

# Constants for quick change
SIZE = 10, 10       # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = 'alamakota'
LIMITS = 0, 0, 0    # Center, Edge, Corner

ITERATIONS = 1000
WINDOW_SIZE = 5, 5


def create_window(player_board, field, windows_size):
    window_x, window_y = windows_size
    window = [['_' for _ in range(window_x)] for _ in range(window_y)]
    half_size_x = window_x // 2
    half_size_y = window_y // 2
    row, col = field

    # for i in range(row - 2, row + 3):
    #    for j in range(col - 2, col + 3):
    #        if 0 <= i <= len(player_board) and 0 <= j <= len(player_board[0]):
    #            window[i][j] = player_board[i][j]

    for i in range(window_y):
        for j in range(window_x):
            # Calculate the actual row and column in the original matrix
            current_row = row - half_size_y + i
            current_col = col - half_size_x + j
            # Check if the calculated row and column are within the bounds of the original matrix
            if 0 <= current_row < len(player_board) and 0 <= current_col < len(player_board[0]):
                window[i][j] = player_board[current_row][current_col]

    return window


def take_input_5x5(size, num_mines, player_board, game_started, filename, window_size):
    size_x, size_y = size
    risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]

    if not game_started:
        row, col = random.randint(0, size_y - 1), random.randint(0, size_x - 1)
    else:
        moves, mines = solve_analytical(player_board)
        risk_board = update_risk_board(num_mines, player_board, risk_board, moves, mines)

        # This part of the code is used to save only the necessary data
        # in order to train neural network, in this case it's a 5x5 window
        # with risked calculated for a central cell
        undiscovered = find_undiscovered_fields(player_board)
        for field, _ in undiscovered:
            row, col = field
            window = create_window(player_board, field, window_size)
            risk = risk_board[row][col]
            save_game_state_5x5(window, risk, filename)

        row, col = choose_least_risky_move(risk_board)
    return row, col, risk_board


def save_game_state_5x5(window, risk, filename):
    with open(filename, 'a') as file:
        for row in window:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')

        file.write('Risk:\n')
        file.write(f'{risk:.2f}\n')

        file.write('\n')


def gameloop_analytical_5x5(size, default_mines, rand_mines, limits, filename, window_size):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        row, col, risk_board = take_input_5x5(size, num_mines, player_board, game_started, filename, window_size)

        if (row, col) == last_input or row is None or col is None:
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        if is_mine(game_board, row, col):
            if not game_started:
                return 'L1'
            return 'L'

        game_started = True

        reveal_squares(game_board, player_board, row, col)
        if is_game_finished(game_board, player_board):
            return 'W'
        

def simulation_5x5(size, default_mines, rand_mines, limits, filename, window_size, seed, iterations):
    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    print('Generating data', end='')
    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'\nProgress: {_}/{iterations}')
        if _ % 5 == 0:
            print('░', end='', flush=True)
        result = gameloop_analytical_5x5(size, default_mines, rand_mines, limits, filename, window_size)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1

    print(f'\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}')
    print('Data saved:', filename)

    filename = filename.replace('.txt', '_summary.txt')
    with open(filename, 'w') as file:
        file.write(f'Wins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}\n')


if __name__ == '__main__':
    os.makedirs('DATASETS', exist_ok=True)
    FILENAME = os.path.join('DATASETS', f'SMP_Data_{SIZE}_{ITERATIONS}.txt')
    start_time = time.time()
    simulation_5x5(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, WINDOW_SIZE, SEED, ITERATIONS)
    print(f'Data generated after: {time.time() - start_time:.2f}s')
