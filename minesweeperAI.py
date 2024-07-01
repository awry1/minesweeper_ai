import os

import numpy as np
import torch
import random
from sympy import *
from minesweeperTrainNN import MinesweeperMLP

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
    print(' ')
    print('+' + '-' * (2 * size - 1) + '+')
    for row in board:
        print('|' + ' '.join(row) + '|')
    print('+' + '-' * (2 * size - 1) + '+')


def print_board_border(board):
    size = len(board)
    print('\n ', end=" ")
    for i in range(size):
        print(str(i), end=" ")
    print(' ')
    print(' +' + '-' * (2 * size - 1) + '+')
    for i, row in enumerate(board):
        print(str(i) + '|' + ' '.join(row) + '|')
    print(' +' + '-' * (2 * size - 1) + '+')


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


def gameloop(size, num_mines, seed):
    game_board, player_board = create_boards(size, num_mines, seed)
    while True:
        print_board_border(player_board)

        row, col = take_input()
        if not is_input_valid(row, col, size):
            print('Invalid input!')
            continue

        if is_mine(game_board, row, col):
            print_end_board(game_board, player_board)
            print('You Lose!')
            quit()

        else:
            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                print_board(player_board)
                print('You Win!')
                quit()


########################## AI part starts here ##########################
def find_undiscovered_fields(size, player_board):
    undiscovered = []
    for row in range(size):
        for col in range(size):
            if not player_board[row][col] == " ":  # or if [row, col] in mines (will this be better?)
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
    n = len(matrix)  # Number of rows in the matrix
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

    return matrix


def find_moves_and_mines(solved_matrix, undiscovered):
    moves = []
    mines = []

    n = len(solved_matrix)
    m = len(solved_matrix[0])

    # Represent undiscovered fields as symbols from x0 to x(m-1)
    hidden_fields = symbols(" ".join(f"x{x}" for x in range(m - 1)), real=True)

    # Create the coefficient matrix (A) and the result vector (b)
    A = Matrix([[solved_matrix[i][j] for j in range(m - 1)] for i in range(n)])
    b = Matrix([solved_matrix[i][m - 1] for i in range(n)])

    # Solve the system of equations
    solution = linsolve((A, b), *hidden_fields)

    i = -1
    for sol in solution:
        for element in sol:
            i += 1
            if element == 1:
                mines.append(undiscovered[i][0])
            elif element == 0:
                moves.append(undiscovered[i][0])
            else:
                continue

    return moves, mines


def solve_analytical(size, player_board):
    undiscovered = find_undiscovered_fields(size, player_board)  # every undiscoverd field with adjacent numbers
    unique_numbers = []  # list of unique adjacent numbers
    for field in undiscovered:
        for number_pos in field[1]:  # undiscovered.neighbours
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
        matrix[i][len(undiscovered)] = ord(
            player_board[number_pos[0]][number_pos[1]]) - 48  # convert number from board to int

    # solve with Gauss elimination
    solved_matrix = solve_gauss(matrix)

    # find moves and mines
    moves, mines = find_moves_and_mines(solved_matrix, undiscovered)

    return moves, mines

# unchanged
def estimate_risk(player_board, row, col):
    total_unknown = 0
    total_mines = 0

    for i in range(-1, 2):
        for j in range(-1, 2):
            x = row + i
            y = col + j
            if 0 <= x < len(player_board) and 0 <= y < len(player_board[0]):
                if player_board[x][y] == ' ':
                    total_unknown += 1
                elif player_board[x][y] > '0':
                    total_mines += ord(player_board[x][y]) - 48

    if total_unknown == 0:
        return 0
    return total_mines / total_unknown

# used to normalize risk to percentage
def find_min_max_risks(size, player_board):
    min_risk = float('inf')
    max_risk = float('-inf')

    for row in range(size):
        for col in range(size):
            if player_board[row][col] == ' ':
                risk = estimate_risk(player_board, row, col)
                if risk < min_risk:
                    min_risk = risk
                if risk > max_risk:
                    max_risk = risk

    return min_risk, max_risk

# modified to normalize each risk and save in a
# risk board - rounded to 2 places
def choose_least_risky_move(size, player_board, risk_board):
    min_risk, max_risk = find_min_max_risks(size, player_board)
    min_normalized_risk = float('inf')
    best_move = None
    current_risk_board = risk_board

    for row in range(size):
        for col in range(size):
            if player_board[row][col] == ' ':
                risk = estimate_risk(player_board, row, col)

                # Normalize the risk
                if max_risk > min_risk:
                    normalized_risk = (risk - min_risk) / (max_risk - min_risk)
                    normalized_risk = round(normalized_risk, 2)
                    if normalized_risk == 0.0:
                        current_risk_board[row][col] = 1.0
                    else:
                        current_risk_board[row][col] = normalized_risk
                else:
                    normalized_risk = 1.0  # If all risks are the same
                    current_risk_board[row][col] = normalized_risk

                if normalized_risk < min_normalized_risk:
                    min_normalized_risk = normalized_risk
                    best_move = (row, col)

    return best_move, current_risk_board


def ai_take_input(size, game_started, player_board):
    moves = []
    risk_board = [[1.0 for _ in range(size)] for _ in range(size)]

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        moves, mines = solve_analytical(size, player_board)
        if len(moves) > 0:
            row, col = moves[0]
        # if no definite moves, choose the least risky move
        else:
            (row, col), risk_board = choose_least_risky_move(size, player_board, risk_board)

    # print('\nAI chose:', col, row)
    return row, col, moves, risk_board


def save_game_state(board, moves, risk_board, filename):
    with open(filename, 'a') as file:
        for row in board:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')

        board_copy = [row.copy() for row in risk_board]
        if len(moves) > 0:
            for move in moves:
                board_copy[move[0]][move[1]] = 0.0

            # Save the modified board state
        file.write('Moves risk factor:\n')
        for row in board_copy:
            row_str = ' '.join([f'{cell:.2f}' for cell in row])
            file.write(row_str + '\n')
        file.write('\n')


def ai_gameloop(size, num_mines, seed, filename):
    game_board, player_board = create_boards(size, num_mines, seed)
    game_started = False
    while True:
        # print_board(player_board)

        row, col, moves, risk_board = ai_take_input(size, game_started, player_board)

        if is_mine(game_board, row, col):
            # print_end_board(game_board, player_board)
            if not game_started:
                # print('\nLose on first')
                # return 'L1'
                continue
            # print('\nLose')
            return 'L'

        else:
            if game_started:
                save_game_state(player_board, moves, risk_board, filename)
            reveal_squares(game_board, player_board, row, col)
            game_started = True
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                # print('\nWin')
                return 'W'


def simulation_ai(size, num_mines, seed, iterations=5000):
    wins, loses, loses1, undecided = 0, 0, 0, 0
    filename = 'trainingData.txt'

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    for _ in range(iterations):
        if _ % 100 == 0:
            print(f"Progress: {_}/{iterations}")
        result = ai_gameloop(size, num_mines, seed, filename)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1

    print(f"\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}")
    quit()


########################## PyTorch part starts here ##########################
def load_model(input_size, hidden_size, output_size):
    model = MinesweeperMLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('minesweeper_nn.pth'))
    model.eval()  # Set the model to evaluation mode
    return model


def convert_board_to_numerical(player_board):
    # Define mapping from characters to numerical values
    mapping = {'0': 0.0, '1': 0.1, '2': 0.2, '3': 0.3, '4': 0.4, '5': 0.5, '6': 0.6, '7': 0.7, '8': 0.8, ' ': 0.9}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 0 for cell in row] for row in player_board]

    return numerical_board


def choose_least_risky_move_torch(size, risk_board):
    best_row, best_col = None, None
    min_risk = float('inf')
    for row in range(size):
        for col in range(size):
            risk = risk_board[row,col]
            if risk < min_risk:
                min_risk = risk
                best_row = row
                best_col = col

    return best_row, best_col

def torch_take_input(size, game_started, player_board, model):
    row, col = None, None

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
            # Convert player_board list to tensor
            true_risk_board = [[1.0 for _ in range(size)] for _ in range(size)]
            true_risk_board = choose_least_risky_move(size, player_board, true_risk_board)
            player_board_numerical = convert_board_to_numerical(player_board)
            player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32).view(1, size * size)

            # Assuming tensor_risk_board is a PyTorch tensor with shape (1, 100)
            tensor_risk_board = model(player_board_tensor)

            # Convert PyTorch tensor to NumPy array
            numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

            # Reshape the NumPy array to represent the board structure (assuming it's a 10x10 board)
            size = 10  # Assuming a square board of size 10x10
            reshaped_board = numpy_risk_board.reshape((size, size))

            # Assuming 'choose_least_risky_move_torch' function expects a size and the risk board
            row, col = choose_least_risky_move_torch(size, reshaped_board)

    # print('\nAI chose:', col, row)
    return row, col


def torch_gameloop(size, num_mines, seed, model):
    game_board, player_board = create_boards(size, num_mines, seed)
    last_input = None
    game_started = False
    while True:
        # print_board(player_board)

        row, col = torch_take_input(size, game_started, player_board, model)
        if not is_input_valid(row, col, size):
            # print('\nUndecided')
            return '?'
        if row is None or col is None or (row, col) == last_input:
            # print('\nUndecided')
            return '?'
        last_input = (row, col)

        if is_mine(game_board, row, col):
            # print_end_board(game_board, player_board)
            if not game_started:
                # print('\nLose on first')
                # return 'L1'
                continue
            # print('\nLose')
            return 'L'

        else:
            reveal_squares(game_board, player_board, row, col)
            if game_started:
                return 'W'
            game_started = True


def simulation_torch(size, num_mines, seed, iterations=1000):
    model = load_model(input_size=(size * size), hidden_size=[600, 400, 200], output_size=(size * size))

    wins, loses, loses1, undecided = 0, 0, 0, 0

    for _ in range(iterations):
        if _ % 100 == 0:
            print(f"Progress: {_}/{iterations}")
        result = torch_gameloop(size, num_mines, seed, model)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1

    print(f"\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}")
    quit()


if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = None

    TORCH = True
    AI = True

    if TORCH:
        simulation_torch(size, num_mines, seed)
    else:
        if AI:
            simulation_ai(size, num_mines, seed)
        else:
            gameloop(size, num_mines, seed)