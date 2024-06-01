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
        row, col = solve_analytical(size, player_board)
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


# AI part
def find_undiscovered_fields(size, player_board):
    undiscovered = []
    for row in range(size):
        for col in range(size):
            if not player_board[row][col] == " ":       # or if [row, col] in mines
                continue
            neighbours = find_adjacent_numbers([row, col], size, player_board)
            if len(neighbours) > 0:
                undiscovered.append([[row, col], neighbours])
    return undiscovered


def find_adjacent_numbers(position, size, player_board):
    numbers = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = position[0] + i
            y = position[1] + j
            if x < 0 or y < 0 or x >= size or y >= size:
                continue
            if player_board[x][y] > "0":
                numbers.append([x, y])
    return numbers


def solve_gauss(matrix):
    # debug print matrix
    print("\nMatrix:")
    for row in matrix:
        print(' '.join(['+' + str(element) if element != 0 else ' _' for element in row]))
    
    n = len(matrix)
    m = len(matrix[0])  # Number of columns in the matrix

    for i in range(min(n, m - 1)):  # Ensure we do not exceed the number of columns
        # Find the pivot row
        pivot_row = i
        while pivot_row < n and matrix[pivot_row][i] == 0:
            pivot_row += 1

        if pivot_row == n:
            continue

        # Swap rows to make the pivot row the current row
        if pivot_row != i:
            matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        # Perform elimination
        for j in range(n):
            if j != i and matrix[j][i] != 0:
                factor = matrix[j][i] // matrix[i][i]
                for k in range(i, m):
                    matrix[j][k] -= factor * matrix[i][k]

    # Normalize rows to handle floating-point issues and to match expected output format
    for i in range(n):
        row_nonzero = next((matrix[i][j] for j in range(m - 1) if matrix[i][j] != 0), None)
        if row_nonzero:
            factor = row_nonzero
            for k in range(m):
                matrix[i][k] //= factor

    # Ensure matrix is in reduced row echelon form
    for i in range(n - 1, -1, -1):
        for j in range(m - 1):
            if matrix[i][j] != 0:
                for k in range(i):
                    factor = matrix[k][j]
                    for l in range(m):
                        matrix[k][l] -= factor * matrix[i][l]
                break
        
    # debug print solved matrix and quit
    print("\nSolved matrix:")
    for row in matrix:
        print(' '.join(['+' + str(element) if element > 0 else str(element) if element < 0 else ' _' for element in row]))
    
    quit()

    return matrix


def solve_analytical(size, player_board):
    undiscovered = find_undiscovered_fields(size, player_board)     # every undiscoverd field with adjacent numbers
    unique_numbers = []     # list of unique adjacent numbers
    for field in undiscovered:
        for number_pos in field[1]:     # undiscovered.neighbours
            if number_pos not in unique_numbers:
                unique_numbers.append(number_pos)

    # create extended matrix to solve and fill last column with adjacent numbers' coordinates
    matrix = [[0 for _ in range(len(undiscovered))] for _ in range(len(unique_numbers))]
    for i in range(len(matrix)):
        matrix[i].append(unique_numbers[i])

    # fill matrix with 1 if undiscovered field is adjacent to number
    for i, x in enumerate(undiscovered):
        for j in range(len(unique_numbers)):
            if matrix[j][len(undiscovered)] in x[1]:
                matrix[j][i] = 1

    # overwrite last column with values instead of coordinates
    for i in range(len(matrix)):
        number_pos = matrix[i][len(undiscovered)]
        matrix[i][len(undiscovered)] = ord(player_board[number_pos[0]][number_pos[1]]) - 48    # convert number from board to int

    # solve with Gauss elimination
    solve_gauss(matrix)

    # find moves and mines
    # append moves
    # append mines

    # return moves

    row = None
    col = None
    return row, col


if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = 10
    game_started = False

    AI = True

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
