from game import *

# Constants for quick change
SIZE = 9, 9         # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0    # Center, Edge, Corner


def gameloop(size, default_mines, rand_mines, seed, limits):
    if seed is not None:
        random.seed(seed)

    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    game_started = False
    while True:
        print_board(player_board)

        row, col = take_input()
        if not is_input_valid(size, row, col):
            print('Invalid input!')
            continue

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)
            game_started = True

        if is_mine(game_board, row, col):
            print_board(game_board)
            print('You Lose!')
            quit()

        reveal_squares(game_board, player_board, row, col)
        if is_game_finished(game_board, player_board):
            print_board(player_board)
            print('You Win!')
            quit()


if __name__ == '__main__':
    gameloop(SIZE, DEFAULT_MINES, RAND_MINES, SEED, LIMITS)
