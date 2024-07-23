import os

import numpy as np
import torch
import random
from sympy import *
from minesweeperTrainNN import MinesweeperMLP

# Constants for quick change
SIZE = 5
DEFAULT_MINES = 4
RAND_MINES = False
SEED = None

HIDDEN_SIZE = [64]
MOVES_LIMIT = 1     # 0 - no limit

TORCH = False
ANALYTICAL = True
SIMPLIFIED = True

ITERATIONS_TORCH = 1000
ITERATIONS_ANALYTICAL = 1000


def random_num_mines(default_mines, rand_mines):
    if not rand_mines:
        return default_mines

    # Max 10, optimized for 4x4 board
    if random.randint(0, 10) < 8:  # 80% chance for less mines
        return random.randint(1, 4)
    return random.randint(4, 11)  # 20% chance for more mines


def create_boards(size, num_mines):
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
    print('\n ', end=' ')
    for i in range(size):
        print(str(i), end=' ')
    print(' ')
    print(' +' + '-' * (2 * size - 1) + '+')
    for i, row in enumerate(board):
        print(str(i) + '|' + ' '.join(row) + '|')
    print(' +' + '-' * (2 * size - 1) + '+')


def take_input():
    col, row = map(int, input('Enter x and y: ').split())
    return row, col


def is_input_valid(size, row, col):
    if row is None or col is None:
        return False
    if row < 0 or row >= size or col < 0 or col >= size:
        return False
    return True


def ensure_fair_start(size, num_mines, game_board, row, col):
    MAX_CENTER = 0
    MAX_EDGE = 0
    MAX_CORNER = 0
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

        if mines_found > MAX_CENTER:
            game_board, player_board = create_boards(size, num_mines)
            needs_check = True
            continue

        if out_of_bounds == 3:  # Edge
            if mines_found > MAX_EDGE:
                game_board, player_board = create_boards(size, num_mines)
                needs_check = True
                continue

        elif out_of_bounds == 5:  # Corner
            if mines_found > MAX_CORNER:
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
    # Prints player_board with mines
    for row in range(len(game_board)):
        for col in range(len(game_board[0])):
            if game_board[row][col] == 'X':
                player_board[row][col] = 'X'
    print_board(player_board)


def gameloop(size, default_mines, rand_mines, seed):
    if seed is not None:
        random.seed(seed)

    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    game_started = False
    while True:
        print_board_border(player_board)

        row, col = take_input()
        if not is_input_valid(size, row, col):
            print('Invalid input!')
            continue

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col)
            game_started = True

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


########################## Analytical Solve part starts here ##########################

##### Gauss Solving #####
def find_undiscovered_fields(player_board):
    undiscovered = []
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':  # or if (row, col) in mines (Will this be better?)
                neighbours = find_adjacent_numbers(player_board, row, col)
                if neighbours:
                    undiscovered.append(((row, col), neighbours))   # ((row, col), (row, col))
    return undiscovered


def find_adjacent_numbers(player_board, row, col):
    numbers = []
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                if player_board[i][j] > '0':
                    numbers.append((i, j))
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
    if m == 2:
        hidden_fields = [symbols(f'x0', real=True)]
    else:
        hidden_fields = symbols(' '.join(f'x{x}' for x in range(m - 1)), real=True)

    # Create the coefficient matrix (A) and the result vector (b)
    A = Matrix([[solved_matrix[i][j] for j in range(m - 1)] for i in range(n)])
    b = Matrix([solved_matrix[i][m - 1] for i in range(n)])

    # Solve the system of equations
    solution = linsolve((A, b), *hidden_fields)

    i = -1
    for sol in solution:
        for element in sol:
            i += 1
            field, neighbours = undiscovered[i]
            if element == 1:
                mines.append(field)
            elif element == 0:
                moves.append(field)
            else:
                continue

    return moves, mines


##### Solving Methods #####
def solve_analytical(player_board):
    undiscovered = find_undiscovered_fields(player_board)  # Every undiscoverd field with adjacent numbers
    unique_numbers = []  # List of unique adjacent numbers
    for field, neighbours in undiscovered:
        for neighbour in neighbours:
            if neighbour not in unique_numbers:
                unique_numbers.append(neighbour)

    # Create extended matrix to solve and fill last column with adjacent numbers' coordinates
    matrix = [[0 for _ in range(len(undiscovered))] for _ in range(len(unique_numbers))]
    for i in range(len(matrix)):
        matrix[i].append(unique_numbers[i])

    # Fill matrix with 1 if undiscovered field is adjacent to number
    for i, x in enumerate(undiscovered):
        for j in range(len(unique_numbers)):
            if matrix[j][len(undiscovered)] in x[1]:
                matrix[j][i] = 1

    # Overwrite last column with values instead of coordinates
    for i in range(len(matrix)):
        number_pos = matrix[i][len(undiscovered)]
        matrix[i][len(undiscovered)] = ord(
            player_board[number_pos[0]][number_pos[1]]) - 48  # Convert number from board to int

    # Solve with Gauss elimination
    solved_matrix = solve_gauss(matrix)

    # Find moves and mines
    moves, mines = find_moves_and_mines(solved_matrix, undiscovered)

    return moves, mines


##### Risk Estimation #####
def estimate_risk(player_board, row, col):
    total_unknown = 0
    total_mines = 0

    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                # it uses unknown neighbors of
                # numbers neighboring the field
                adjacent_numbers = len(find_adjacent_numbers(player_board, i, j))
                if player_board[i][j] == ' ' and adjacent_numbers > 0:
                    total_unknown += 1
                elif player_board[i][j] > '0':
                    total_mines += ord(player_board[i][j]) - 48
            else:  # If out of bounds (Is this needed?)
                total_unknown += 1

    if total_unknown == 0:
        return 0
    return total_mines / total_unknown


def find_min_max_risks(player_board):
    # Used to normalize risk values
    min_risk = float('inf')
    max_risk = float('-inf')

    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':
                risk = estimate_risk(player_board, row, col)
                if risk < min_risk:
                    min_risk = risk
                if risk > max_risk:
                    max_risk = risk

    return min_risk, max_risk


def find_normalized_risk(player_board, row, col, min_risk, max_risk):
    risk = estimate_risk(player_board, row, col)
    if risk == 0.0:
        return 1.0

    if max_risk > min_risk:
        normalized_risk = (risk - min_risk) / (max_risk - min_risk)
        normalized_risk = round(normalized_risk, 2)
        return normalized_risk
    return 1.0  # If all risks are the same


def update_risk_board(num_mines, player_board, risk_board, moves, mines):
    """ # Don't know if works as intended
    if len(mines) == num_mines:
        for row in range(len(player_board)):
            for col in range(len(player_board[0])):
                if player_board[row][col] == ' ' and (row, col) not in mines:
                    risk_board[row][col] = 0.0
        return risk_board """

    min_risk, max_risk = find_min_max_risks(player_board)
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':
                if (row, col) in moves:
                    risk_board[row][col] = 0.0
                # Already 1.0 on the board, no need to update
                # elif (row, col) in mines:
                #     risk_board[row][col] = 1.0
                else:
                    if (row, col) in mines:
                        risk_board[row][col] = 1.0
                    else:
                        risk_board[row][col] = find_normalized_risk(player_board, row, col, min_risk, max_risk)
    return risk_board


def choose_least_risky_move(risk_board):
    min_risk = float('inf')
    best_move = None

    for row in range(len(risk_board)):
        for col in range(len(risk_board[0])):
            if risk_board[row][col] < min_risk:
                min_risk = risk_board[row][col]
                best_move = row, col

    return best_move


##### Input Methods #####
def create_window(player_board, field):
    window = [[' ' for _ in range(5)] for _ in range(5)]
    half_size = 5 // 2
    row, col = field

    #for i in range(row-2, row+3):
    #    for j in range(col-2, col+3):
    #        if 0 <= i <= len(player_board) and 0 <= j <= len(player_board[0]):
    #            window[i][j] = player_board[i][j]

    for i in range(5):
        for j in range(5):
            # Calculate the actual row and column in the original matrix
            current_row = row - half_size + i
            current_col = col - half_size + j
            # Check if the calculated row and column are within the bounds of the original matrix
            if 0 <= current_row < len(player_board) and 0 <= current_col < len(player_board[0]):
                window[i][j] = player_board[current_row][current_col]

    return window

# For window solving approach (simplified)
def take_input_5x5(size, num_mines, player_board, game_started, filename):
    risk_board = [[1.0 for _ in range(size)] for _ in range(size)]
    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        undiscovered = find_undiscovered_fields(player_board)
        for field, _ in undiscovered:
            row, col = field
            window = create_window(player_board, field)
            moves, mines = solve_analytical(window)
            if (2, 2) in moves:
                risk = 0.0
            elif (2, 2) in mines:
                risk = 1.0
            else:
                min_risk, max_risk = find_min_max_risks(player_board)
                risk = find_normalized_risk(window, row, col, min_risk, max_risk)
            save_game_state_5x5(window, risk, filename)

            risk_board[row][col] = risk
        row, col = choose_least_risky_move(risk_board)
    return row, col, risk_board


# For full board solving approach (default)
def take_input_analytical(size, num_mines, player_board, game_started):
    risk_board = [[1.0 for _ in range(size)] for _ in range(size)]

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        moves, mines = solve_analytical(player_board)
        risk_board = update_risk_board(num_mines, player_board, risk_board, moves, mines)

        if moves:
            row, col = moves[0]
        # If no definite moves, choose the least risky move
        else:
            row, col = choose_least_risky_move(risk_board)

    # print('\nAI chose:', col, row)
    return row, col, risk_board

##### Game Saves #####
def save_game_state_5x5(window, risk, filename):
    with open(filename, 'a') as file:
        for row in window:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')
        file.write('Risk:\n')
        file.write(f'{risk}\n')
        file.write('\n')


def save_game_state(board, risk_board, filename):

    with open(filename, 'a') as file:
        for row in board:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')

        file.write('Risk factors:\n')
        for row in risk_board:
            row_str = ' '.join([f'{cell:.2f}' for cell in row])
            file.write(row_str + '\n')

        file.write('\n')

##### Simulation #####
def gameloop_analytical(size, default_mines, rand_mines, filename):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        # print_board(player_board)

        # Defines a type of board solving - moving window or default (entire board)
        if SIMPLIFIED:
            row, col, risk_board = take_input_5x5(size, num_mines, player_board, game_started, filename)
        else:
            row, col, risk_board = take_input_analytical(size, num_mines, player_board, game_started)

        if (row, col) == last_input or row is None or col is None:
            return '?'
        
        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col)

        last_input = row, col

        if is_mine(game_board, row, col):
            # print_end_board(game_board, player_board)
            if not game_started:
                return 'L1'
            return 'L'

        else:
            if game_started and not SIMPLIFIED:
                save_game_state(player_board, risk_board, filename)
            game_started = True

            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                return 'W'


def simulation_analytical(size, default_mines, rand_mines, filename, iterations, seed):
    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'Progress: {_}/{iterations}')
        result = gameloop_analytical(size, default_mines, rand_mines, filename)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1

    print(f'\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}')
    quit()


########################## PyTorch part starts here ##########################
def load_model(input_size, hidden_size, output_size, filename):
    model = MinesweeperMLP(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    return model


def convert_board_to_numerical(board):
    # Define mapping from characters to numerical values
    mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, ' ': 9.0}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 0 for cell in row] for row in board]

    return numerical_board


def save_torch_results(true_risk_board, torch_num_mines, torch_risk_board, filename):
    with open(filename, 'a') as file:
        file.write('True risk factors:\n')

        for row in true_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('Torch risk factors:\n')

        for row in torch_risk_board:
            row_str = ' '.join([f'{round(cell, 3):.3f}' for cell in row])
            file.write(row_str + '\n')

        file.write('Torch number of mines:\n')
        file.write(str(torch_num_mines) + '\n')
        file.write('\n')


def take_input_torch(size, num_mines, player_board, game_started, model, filename):
    row, col = None, None

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
        return row, col

    with torch.no_grad():  # Disable gradient computation during evaluation
        # Convert player_board list to tensor
        player_board_numerical = convert_board_to_numerical(player_board)
        player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32).view(1, size * size)

        # For debugging purposes
        risk_board = [[1.0 for _ in range(size)] for _ in range(size)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])

        tensor_risk_board = model(player_board_tensor)

        # Convert PyTorch tensor to NumPy array
        numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

        # Reshape the NumPy array to represent the board structure
        reshaped_board = numpy_risk_board.reshape((size, size))

        save_torch_results(risk_board, num_mines, reshaped_board, filename)

        row, col = choose_least_risky_move(reshaped_board)
    
    if not is_input_valid(size, row, col):
        return None, None

    # print('\nAI chose:', col, row)
    return row, col


def gameloop_torch(size, default_mines, rand_mines, filename, model, moves_limit):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        # print_board(player_board)
        risk_board = [[1.0 for _ in range(size)] for _ in range(size)]
        risk_board = update_risk_board(num_mines, player_board, risk_board, [], [])
        true_row, true_col = choose_least_risky_move(risk_board)

        row, col = take_input_torch(size, num_mines, player_board, game_started, model, filename)
        
        if (row, col) == last_input or row is None or col is None:
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col)

        last_input = row, col

        if game_started:
            if true_row != row or true_col != col or is_mine(game_board, row, col):
                # print_end_board(game_board, player_board)
                if not game_started:
                    return 'L1'
                return 'L'

        else:
            if game_started:
                moves_limit -= 1
                if moves_limit == 0:
                    return 'W'
            game_started = True

            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                return 'W'


def simulation_torch(size, default_mines, rand_mines, filename, model_filename, hidden_size, moves_limit, iterations, seed):
    model = load_model((size * size), hidden_size, (size * size), model_filename)

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'Progress: {_}/{iterations}')
        result = gameloop_torch(size, default_mines, rand_mines, filename, model, moves_limit)
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

########################## Main starts here ##########################
if __name__ == '__main__':
    if TORCH:
        # Used to evaluate the model
        MODEL_FILENAME = os.path.join('MODELS', f'Model_{SIZE}.pth')
        os.makedirs('RESULTS_TEST', exist_ok=True)
        FILENAME = os.path.join('RESULTS_TEST', f'TestResult_{SIZE}_{ITERATIONS_TORCH}.txt')
        simulation_torch(SIZE, DEFAULT_MINES, RAND_MINES, FILENAME, MODEL_FILENAME, HIDDEN_SIZE, MOVES_LIMIT, ITERATIONS_TORCH, SEED)
    else:
        if ANALYTICAL:
            # Used to generate training data
            os.makedirs('DATA', exist_ok=True)
            if SIMPLIFIED:
                FILENAME = os.path.join('DATA', f'SMP_Data_{SIZE}_{ITERATIONS_ANALYTICAL}.txt')
            else:
                FILENAME = os.path.join('DATA', f'Data_{SIZE}_{ITERATIONS_ANALYTICAL}.txt')
            simulation_analytical(SIZE, DEFAULT_MINES, RAND_MINES, FILENAME, ITERATIONS_ANALYTICAL,SEED)
        else:
            # Play the game manually
            gameloop(SIZE, DEFAULT_MINES, RAND_MINES, SEED)
