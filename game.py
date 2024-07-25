import random

def random_num_mines(default_mines, rand_mines):
    if not rand_mines:
        return default_mines
    
    # Max 10, optimized for 5x5 board
    if random.randint(0, 10) < 8:  # 80% chance for less mines
        return random.randint(1, 4)
    return random.randint(4, 11)  # 20% chance for more mines


def create_boards(size, num_mines):
    size_x, size_y = size
    game_board = [['0' for _ in range(size_x)] for _ in range(size_y)]
    player_board = [[' ' for _ in range(size_x)] for _ in range(size_y)]

    mines = random.sample(range(size_x * size_y), num_mines)
    for i in mines:
        row = i // size_x
        col = i % size_y
        game_board[row][col] = 'X'

    for row in range(size_y):
        for col in range(size_x):
            if game_board[row][col] == '0':
                game_board[row][col] = str(count_adjacent_mines(game_board, row, col))

    return game_board, player_board


def count_adjacent_mines(game_board, row, col):
    count = 0
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                if game_board[i][j] == 'X':
                    count += 1
    return count


def print_board(board):
    size_x = len(board[0])
    print()

    # Print column headers
    print('  ' + ' '.join(map(str, range(size_x))) + ' ')

    # Print top border
    print(' +' + '-' * (2 * size_x - 1) + '+')

    # Print each row with row headers and borders
    for i, row in enumerate(board):
        print(str(i) + '|' + ' '.join(row) + '|')

    # Print bottom border
    print(' +' + '-' * (2 * size_x - 1) + '+')


def print_board2(board):
    size_x = len(board[0])
    print()

    # Print column headers
    print(' |' + '|'.join(map(str, range(size_x))) + '|')

    # Print top border
    print('-' + '+-' * size_x + '-')

    # Print each row with row headers and borders
    for i, row in enumerate(board):
        print(f'{i}|' + '|'.join(row) + '|')
        print('-' + '+-' * size_x + '+')


def take_input():
    col, row = map(int, input('Enter x and y: ').split())
    return row, col


def is_input_valid(size, row, col):
    size_x, size_y = size
    if row is None or col is None:
        return False
    if row < 0 or row >= size_y or col < 0 or col >= size_x:
        return False
    return True


def ensure_fair_start(size, num_mines, game_board, row, col, limits):
    max_center, max_edge, max_corner = limits
    needs_check = True

    while needs_check:
        needs_check = False
        mines_found = 0
        out_of_bounds = 0

        # Ensure first move is not a mine
        if game_board[row][col] == 'X':
            game_board, player_board = create_boards(size, num_mines)
            needs_check = True
            continue

        # Check amount of neighboring mines
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                    if game_board[i][j] == 'X':
                        mines_found += 1
                else:
                    out_of_bounds += 1

        if mines_found > max_center:
            game_board, player_board = create_boards(size, num_mines)
            needs_check = True
            continue

        if out_of_bounds == 3:  # Edge
            if mines_found > max_edge:
                game_board, player_board = create_boards(size, num_mines)
                needs_check = True
                continue

        elif out_of_bounds == 5:  # Corner
            if mines_found > max_corner:
                game_board, player_board = create_boards(size, num_mines)
                needs_check = True
                continue

    return game_board


def is_mine(game_board, row, col):
    if game_board[row][col] == 'X':
        return True
    return False


def reveal_squares(game_board, player_board, row, col):
    if player_board[row][col] != ' ':
        return
    # Not possible to enter function if mine
    # if game_board[row][col] == 'X':
    #     return
    player_board[row][col] = game_board[row][col]
    if player_board[row][col] == '0':
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                    reveal_squares(game_board, player_board, i, j)


def is_game_finished(game_board, player_board):
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] != 'X' and player_board[row][col] == ' ':
                return False
    return True


if __name__ == '__main__':
    print('solve_human.py          play the game')
    print('solve_analytical.py     create training data')
    print('train_nn.py             train neural network')
    print('solve_nn.py             test neural network')


# Old code
def create_boards(size, num_mines):
    size_x, size_y = size
    game_board = [[' ' for _ in range(size_x)] for _ in range(size_y)]
    player_board = [[' ' for _ in range(size_x)] for _ in range(size_y)]

    mines = random.sample(range(size_x * size_y), num_mines)
    for i in mines:
        row = i // size_x
        col = i % size_y
        game_board[row][col] = 'X'

    return game_board, player_board


def reveal_squares(game_board, player_board, row, col):
    if player_board[row][col] != ' ':
        return
    # Not possible to enter function if mine
    # if game_board[row][col] == 'X':
    #     return
    player_board[row][col] = str(count_adjacent_mines(game_board, row, col))
    if player_board[row][col] == '0':
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                    reveal_squares(game_board, player_board, i, j)