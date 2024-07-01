import os
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


########################## Analytical Solve part starts here ##########################
def find_undiscovered_fields(size, player_board):
    undiscovered = []
    for row in range(size):
        for col in range(size):
            if not player_board[row][col] == " ":  # or if [row, col] in mines (Will this be better?)
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
    undiscovered = find_undiscovered_fields(size, player_board)  # Every undiscoverd field with adjacent numbers
    unique_numbers = []  # List of unique adjacent numbers
    for field in undiscovered:
        for number_pos in field[1]:  # Undiscovered.neighbours
            if number_pos not in unique_numbers:
                unique_numbers.append(number_pos)

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


def estimate_risk(player_board, row, col):
    total_unknown = 0
    total_mines = 0

    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                if player_board[i][j] == ' ':
                    total_unknown += 1
                elif player_board[i][j] > '0':
                    total_mines += ord(player_board[i][j]) - 48
            # else:   # If out of bounds
            #     total_unknown += 1

    if total_unknown == 0:
        return 0
    return total_mines / total_unknown


def find_min_max_risks(size, player_board):
    # Used to normalize risk values
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


def find_normalized_risk(player_board, row, col, min_risk, max_risk):
    risk = estimate_risk(player_board, row, col)
    if max_risk > min_risk:
        normalized_risk = (risk - min_risk) / (max_risk - min_risk)
        normalized_risk = round(normalized_risk, 2)
        if normalized_risk == 0.0:
            return 1.0
        return normalized_risk
    return 1.0  # If all risks are the same


def update_risk_board(size, player_board, risk_board, moves, mines):
    min_risk, max_risk = find_min_max_risks(size, player_board)
    for row in range(size):
        for col in range(size):
            if player_board[row][col] == ' ':
                if (row, col) in moves:
                    risk_board[row][col] = 0.0
                elif (row, col) in mines:
                    risk_board[row][col] = 1.0
                else:
                    risk_board[row][col] = find_normalized_risk(player_board, row, col, min_risk, max_risk)
    return risk_board


def choose_least_risky_move(size, risk_board):
    min_risk = float('inf')
    best_move = None

    for row in range(size):
        for col in range(size):
            if risk_board[row][col] < min_risk:
                min_risk = risk_board[row][col]
                best_move = row, col

    return best_move


def take_input_analytical(size, game_started, player_board):
    moves = []
    risk_board = [[1.0 for _ in range(size)] for _ in range(size)]

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        moves, mines = solve_analytical(size, player_board)
        risk_board = update_risk_board(size, player_board, risk_board, moves, mines)

        if len(moves) > 0:
            row, col = moves[0]
        # If no definite moves, choose the least risky move
        else:
            row, col = choose_least_risky_move(size, risk_board)

    # print('\nAI chose:', col, row)
    return row, col, risk_board


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


def gameloop_analytical(size, num_mines, seed, filename):
    game_board, player_board = create_boards(size, num_mines, seed)
    game_started = False
    while True:
        # print_board(player_board)

        row, col, risk_board = take_input_analytical(size, game_started, player_board)

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
                save_game_state(player_board, risk_board, filename)
            reveal_squares(game_board, player_board, row, col)
            game_started = True
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                # print('\nWin')
                return 'W'


def simulation_analytical(size, num_mines, seed, iterations=10000):
    filename = 'trainingData.txt'

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f"Progress: {_}/{iterations}")
        result = gameloop_analytical(size, num_mines, seed, filename)
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


def convert_board_to_numerical(board):
    # Define mapping from characters to numerical values
    mapping = {'0': 0.0, '1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '6': 6.0, '7': 7.0, '8': 8.0, ' ': 9.0}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 0 for cell in row] for row in board]

    return numerical_board


def take_input_torch(size, game_started, player_board, model):
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
            risk_board = update_risk_board(size, player_board, risk_board, [], [])

            tensor_risk_board = model(player_board_tensor)

            # Convert PyTorch tensor to NumPy array
            numpy_risk_board = tensor_risk_board.cpu().detach().numpy()

            # Reshape the NumPy array to represent the board structure
            reshaped_board = numpy_risk_board.reshape((size, size))

            row, col = choose_least_risky_move(size, reshaped_board)

    # print('\nAI chose:', col, row)
    return row, col


def gameloop_torch(size, num_mines, seed, model):
    game_board, player_board = create_boards(size, num_mines, seed)
    last_input = None
    game_started = False
    move_count = 0
    while True:
        # print_board(player_board)

        row, col = take_input_torch(size, game_started, player_board, model)
        # If working correctly, there should be no need to check if input is valid/last input (?)
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
            move_count += 1
            game_started = True
            reveal_squares(game_board, player_board, row, col)
            # Limit the number of moves
            if move_count >= 3:
                return 'W'
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                # print('\nWin')
                return 'W'


def simulation_torch(size, num_mines, seed):
    iterations = 1000
    hidden_size = [1000, 1000]

    model = load_model(input_size=(size * size), hidden_size=hidden_size, output_size=(size * size))

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f"Progress: {_}/{iterations}")
        result = gameloop_torch(size, num_mines, seed, model)
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


########################## main starts here ##########################
if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = None

    TORCH = True
    ANALYTICAL = True

    if TORCH:
        # Used to evaluate the model
        simulation_torch(size, num_mines, seed)
    else:
        if ANALYTICAL:
            # Used to generate training data
            simulation_analytical(size, num_mines, seed)
        else:
            # Play the game manually
            gameloop(size, num_mines, seed)
