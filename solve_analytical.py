from game import *
from sympy import Matrix, symbols, linsolve
import os

# Constants for quick change
SIZE = 10, 10   # X, Y
DEFAULT_MINES = 10
RAND_MINES = False
SEED = None
LIMITS = 0, 0, 0    # Center, Edge, Corner

ITERATIONS = 1000


def find_undiscovered_fields(player_board):
    undiscovered = []
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':
                numbers = find_adjacent_numbers(player_board, row, col)
                if numbers:
                    undiscovered.append(((row, col), numbers))  # ((row, col), (row, col))
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
    n = len(matrix) # Number of rows in the matrix
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


def solve_analytical(player_board):
    undiscovered = find_undiscovered_fields(player_board)   # Every undiscoverd field with adjacent numbers
    unique_numbers = [] # List of unique adjacent numbers
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
            player_board[number_pos[0]][number_pos[1]]) - 48    # Convert number from board to int

    # Solve with Gauss elimination
    solved_matrix = solve_gauss(matrix)

    # Find moves and mines
    moves, mines = find_moves_and_mines(solved_matrix, undiscovered)

    return moves, mines


def estimate_risk(player_board, row, col):
    # Find all adjacent numbers and store them in list
    numbers = find_adjacent_numbers(player_board, row, col)

    # For every number in list find number of adjacent hidden fields
    # Calculate subrisk and store it in list (subrisk = value / hidden_fields)
    risks = []
    for number in numbers:
        row, col = number
        hidden_fields = 0
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                    if player_board[i][j] == ' ':
                        hidden_fields += 1
        # # If somehow the number is not adjacent EVEN TO THE GIVEN FIELD
        # if hidden_fields == 0:
        #     risk.append(0)
        #     continue
        value = ord(player_board[row][col]) - 48
        risks.append((value / hidden_fields))

    # Estimated risk is sum of every subrisk calculated above
    risk = 0
    for subrisk in risks:
        risk += subrisk
    return risk


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
    # # Don't know if works as intended
    # if len(mines) == num_mines:
    #     for row in range(len(player_board)):
    #         for col in range(len(player_board[0])):
    #             if player_board[row][col] == ' ' and (row, col) not in mines:
    #                 risk_board[row][col] = 0.0
    #     return risk_board

    min_risk, max_risk = find_min_max_risks(player_board)
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':
                if (row, col) in moves:
                    risk_board[row][col] = 0.0
                # # Already 1.0 on the board, no need to update
                # elif (row, col) in mines:
                #     risk_board[row][col] = 1.0
                else:
                    if (row, col) in mines:
                        risk_board[row][col] = 1.0
                    else:
                        risk_board[row][col] = find_normalized_risk(player_board, row, col, min_risk, max_risk)
    return risk_board


def update_risk_board2(player_board, risk_board, moves, mines):
    # UNTESTED!!!
    # Find all numbers on the board
    numbers = []
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] > '0':
                numbers.append((row, col))
    
    # For every number in list find number of neighbouring hidden fields
    # Calculate subrisk and store it in list (subrisk = value / hidden_fields)
    risks = []
    for number in numbers:
        row, col = number
        hidden_fields = 0
        value = ord(player_board[row][col]) - 48
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                    if player_board[i][j] == ' ':
                        hidden_fields += 1
        if hidden_fields > 0:
            risks.append(((row, col), (value / hidden_fields)))
        
    # For every hidden field calculate risk based on neighbouring numbers
    for number, subrisk in risks:
        row, col = number
        for i in range(row - 1, row + 2):
            for j in range(col - 1, col + 2):
                if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]):
                    if player_board[i][j] == ' ':
                        risk_board[i][j] -= subrisk
    
    # Fix the risk values and fill in the moves
    for row in range(len(player_board)):
        for col in range(len(player_board[0])):
            if player_board[row][col] == ' ':
                if (row, col) in moves:
                    risk_board[row][col] = 0.0
                    continue
                # # Already 1.0 on the board, no need to update
                # if (row, col) in mines:
                #     risk_board[row][col] = 1.0
                #     continue
                if risk_board[row][col] != 1.0:
                    risk_board[row][col] -= 1.0
                    risk_board[row][col] *= -1.0
    
    # Normalize the risk values
    min_risk = min([min(row) for row in risk_board])
    max_risk = max([max(row) for row in risk_board])
    for row in range(len(risk_board)):
        for col in range(len(risk_board[0])):
            if risk_board[row][col] != 1.0 and risk_board[row][col] != 0.0:
                risk_board[row][col] = (risk_board[row][col] - min_risk) / (max_risk - min_risk)

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


def take_input_analytical(size, num_mines, player_board, game_started):
    size_x, size_y = size
    risk_board = [[1.0 for _ in range(size_x)] for _ in range(size_y)]

    if not game_started:
        row, col = random.randint(0, size_y - 1), random.randint(0, size_x - 1)
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


def gameloop_analytical(size, default_mines, rand_mines, limits, filename):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    last_input = None
    game_started = False
    while True:
        row, col, risk_board = take_input_analytical(size, num_mines, player_board, game_started)

        if (row, col) == last_input or row is None or col is None:
            return '?'
        
        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col, limits)

        last_input = row, col

        if is_mine(game_board, row, col):
            if not game_started:
                return 'L1'
            return 'L'

        else:
            if game_started:
                save_game_state(player_board, risk_board, filename)
            game_started = True

            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                return 'W'


def simulation(size, default_mines, rand_mines, limits, filename, seed, iterations):
    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(iterations):
        if _ % 100 == 0:
            print(f'Progress: {_}/{iterations}')
        result = gameloop_analytical(size, default_mines, rand_mines, limits, filename)
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


if __name__ == '__main__':
    os.makedirs('DATA', exist_ok=True)
    FILENAME = os.path.join('DATA', f'Data_{SIZE}_{ITERATIONS}.txt')
    simulation(SIZE, DEFAULT_MINES, RAND_MINES, LIMITS, FILENAME, SEED, ITERATIONS)