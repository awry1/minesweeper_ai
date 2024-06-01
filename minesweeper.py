import random


def create_boards(size, num_mines, seed):
    if seed is not None:
        random.seed(seed)

    game_board = [[' ' for _ in range(size)] for _ in range(size)]
    player_board = [[' ' for _ in range(size)] for _ in range(size)]

    mines = random.sample(range(size * size), num_mines)
    for i in mines:
        row = i // size
        col = i % size
        game_board[row][col] = 'X'

    return game_board, player_board


def print_board(board):
    size = len(board)
    print(' ', end = " ")
    for i in range(size):
        print(str(i), end = " ")
    print(' ')
    print(' +' + '-' * (2 * size - 1) + '+')
    row_number = 0
    for row in board:
        print(str(row_number) + '|' + ' '.join(row) + '|')
        row_number += 1
    print(' +' + '-' * (2 * size - 1) + '+')


def ai_take_input(size, game_started, player_board):
    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
        # row, col = solve_analytical(size, player_board)
        # if len(moves) == 0:
        #   moves = solve_analytical(size, player_board)
        # row, col = moves.pop

    print('AI chose:', col, row, "\n")
    return row, col


def take_input():
    col, row = map(int, input('Enter x and y: ').split())
    return row, col


def is_input_valid(row, col, size):
    if 0 <= row < size and 0 <= col < size:
        return True
    return False


def is_mine(game_board, row, col):
    if game_board[row][col] == 'X':
        return True
    return False


def reveal_squares(game_board, player_board, row, col):
    if player_board[row][col] != ' ':
        return
    if game_board[row][col] == 'X':
        return
    player_board[row][col] = str(count_adjacent_mines(game_board, row, col))
    if player_board[row][col] == '0':
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                    reveal_squares(game_board, player_board, i, j)


def count_adjacent_mines(game_board, row, col):
    count = 0
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                if game_board[i][j] == 'X':
                    count += 1
    return count


def is_game_finished(game_board, player_board):
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] != 'X' and player_board[row][col] == ' ':
                return False
    return True


def print_end_board(game_board, player_board):
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] == 'X':
                player_board[row][col] = 'X'
    print_board(player_board)


if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = 10
    game_started = False

    AI = False

    game_board, player_board = create_boards(size, num_mines, seed)
    while True:
        print_board(player_board)

        if AI:
            row, col = ai_take_input(size, game_started, player_board)
            if not game_started:
                game_started = True
        else:
            row, col = take_input()
            if not is_input_valid(row, col, size):
                print('Invalid input!')
                continue

        if is_mine(game_board, row, col):
            print_end_board(game_board, player_board)
            print('You lose!')
            break

        else:
            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                print_end_board(game_board, player_board)
                print('You win!')
                break
