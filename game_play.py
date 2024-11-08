# This file shows examle of solved game using analytical method

from game import *
from _1_solve_analytical import solve_analytical, choose_least_risky_move, update_risk_board
from _1_solve_analytical_5x5 import create_window

# Constants for quick change
SIZE = 10, 10       # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = 'alamakota'
LIMITS = 0, 0, 0    # Center, Edge, Corner


def take_input_5x5(size, num_mines, player_board, game_started):
    size_x, size_y = size
    risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]

    if not game_started:
        row, col = random.randint(0, size_y - 1), random.randint(0, size_x - 1)
    else:
        moves, mines = solve_analytical(player_board)
        risk_board = update_risk_board(num_mines, player_board, risk_board, moves, mines)

        row, col = choose_least_risky_move(risk_board)

    return row, col


def gameloop(size, default_mines, rand_mines, limits):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        print_board(player_board)
        row, col = take_input_5x5(size, num_mines, player_board, game_started)

        if (row, col) == last_input or row is None or col is None:
            print('Unable to make a move!')
            return
        else:
            print('Move:', row, col)

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        if is_mine(game_board, row, col):
            if not game_started:
                print('Lose on first move!')
                return
            print('Lose!')
            return

        game_started = True

        reveal_squares(game_board, player_board, row, col)
        if is_game_finished(game_board, player_board):
            print_board(player_board)
            print('Win!')
            return
        

if __name__ == '__main__':
    # random.seed(SEED)
    gameloop(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS)
