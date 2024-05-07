import random

def create_boards(size, num_mines, seed):
    if seed is not None:
        random.seed(seed)

    game_board = [[' ' for _ in range(size)] for _ in range(size)]
    player_board = [[' ' for _ in range(size)] for _ in range(size)]

    mines = random.sample(range(size*size), num_mines)
    for i in mines:
        row = i // size
        col = i % size
        game_board[row][col] = 'X'

    return game_board, player_board

def print_board(board):
    size = len(board)
    print('+' + '-' * (2*size-1) + '+')
    for row in board:
        print('|' + ' '.join(row) + '|')
    print('+' + '-' * (2*size-1) + '+')

def reveal_squares(game_board, player_board, row, col):
    if player_board[row][col] != ' ':
        return
    if game_board[row][col] == 'X':
        return
    player_board[row][col] = str(count_adjacent_mines(game_board, row, col))
    if player_board[row][col] == '0':
        for i in range(row-1, row+2):
            for j in range(col-1, col+2):
                if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                    reveal_squares(game_board, player_board, i, j)

def count_adjacent_mines(game_board, row, col):
    count = 0
    for i in range(row-1, row+2):
        for j in range(col-1, col+2):
            if 0 <= i < len(game_board) and 0 <= j < len(game_board[0]):
                if game_board[i][j] == 'X':
                    count += 1
    return count

def is_game_over(game_board, player_board):
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] != 'X' and player_board[row][col] == ' ':
                return False
    return True

def take_input():
    row, col = map(int, input('Enter x and y: ').split())
    return row, col

def ai_take_input(player_board, size):
    # Some day I will implement a real AI
    row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    print('AI chose:', row, col)
    return row, col

def check_input(row, col, size):
    if (0 <= row < size and 0 <= col < size):
        return True
    return False

def check_mines(game_board, row, col):
    if game_board[row][col] == 'X':
        return True
    return False

def print_end_board(game_board, player_board):
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] == 'X':
                player_board[row][col] = 'X'
    print_board(player_board)

if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = None

    human = True

    game_board, player_board = create_boards(size, num_mines, seed)
    while True:
        print_board(player_board)

        if not human:
            row, col = ai_take_input(player_board, size)
        else:
            row, col = take_input()
            if not check_input(row, col, size):
                print('Invalid input!')
                continue

        if check_mines(game_board, row, col):
            print_end_board(game_board, player_board)
            print('You lose!')
            break

        else:
            reveal_squares(game_board, player_board, row, col)
            if is_game_over(game_board, player_board):
                print_end_board(game_board, player_board)
                print('You win!')
                break