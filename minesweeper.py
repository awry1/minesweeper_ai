import random

import torch
from sympy import *

from minesweeperTrainNN import MinesweeperNN


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
    return 0 <= row < size and 0 <= col < size

def is_mine(game_board, row, col):
    return game_board[row][col] == 'X'

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
            if not player_board[row][col] == " ":
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
    n = len(matrix)
    m = len(matrix[0])

    for i in range(min(n, m - 1)):
        pivot_row = i
        while pivot_row < n and matrix[pivot_row][i] == 0:
            pivot_row += 1

        if pivot_row == n:
            continue

        if pivot_row != i:
            matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        for j in range(n):
            if j != i and matrix[j][i] != 0:
                factor = matrix[j][i] // matrix[i][i]
                for k in range(i, m):
                    matrix[j][k] -= factor * matrix[i][k]

    for i in range(n):
        row_nonzero = next((matrix[i][j] for j in range(m - 1) if matrix[i][j] != 0), None)
        if row_nonzero:
            factor = row_nonzero
            for k in range(m):
                matrix[i][k] //= factor

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

    hidden_fields = symbols(" ".join(f"x{x}" for x in range(m - 1)), real=True)

    A = Matrix([[solved_matrix[i][j] for j in range(m - 1)] for i in range(n)])
    b = Matrix([solved_matrix[i][m - 1] for i in range(n)])

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
    undiscovered = find_undiscovered_fields(size, player_board)
    unique_numbers = []
    for field in undiscovered:
        for number_pos in field[1]:
            if number_pos not in unique_numbers:
                unique_numbers.append(number_pos)

    matrix = [[0 for _ in range(len(undiscovered))] for _ in range(len(unique_numbers))]
    for i in range(len(matrix)):
        matrix[i].append(unique_numbers[i])

    for i, x in enumerate(undiscovered):
        for j in range(len(unique_numbers)):
            if matrix[j][len(undiscovered)] in x[1]:
                matrix[j][i] = 1

    for i in range(len(matrix)):
        number_pos = matrix[i][len(undiscovered)]
        matrix[i][len(undiscovered)] = ord(player_board[number_pos[0]][number_pos[1]]) - 48

    solved_matrix = solve_gauss(matrix)

    moves, mines = find_moves_and_mines(solved_matrix, undiscovered)

    return moves, mines

def ai_take_input_analytical(size, game_started, player_board):
    row, col = None, None

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        moves, mines = solve_analytical(size, player_board)
        if len(moves) > 0:
            row, col = moves[0]
        else:
            row, col = choose_least_risky_move(size, player_board)

    #print('\nAI chose:', col, row)
    return row, col

def choose_least_risky_move(size, player_board):
    min_risk = float('inf')
    best_move = None

    for row in range(size):
        for col in range(size):
            if player_board[row][col] == ' ':
                risk = estimate_risk(player_board, row, col)
                if risk < min_risk:
                    min_risk = risk
                    best_move = (row, col)

    return best_move

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

def save_game_state(board, move, filename):
    with open(filename, 'a') as file:
        for row in board:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')
        file.write(f"{move[0]} {move[1]}\n")

def ai_gameloop_analytical(size, num_mines, seed):
    game_board, player_board = create_boards(size, num_mines, seed)
    game_started = False
    while True:
        row, col = ai_take_input_analytical(size, game_started, player_board)
        if row is None or col is None:
            #print('\nUndecided')
            return '?'

        if is_mine(game_board, row, col):
            if not game_started:
                #print('\nLose on first')
                return 'L1'
            #print('\nLose')
            return 'L'

        else:
            reveal_squares(game_board, player_board, row, col)
            if game_started:
                save_game_state(player_board, (row, col), 'trainingData.txt')
            game_started = True
            if is_game_finished(game_board, player_board):
                #print('\nWin')
                return 'W'

def simulation(size, num_mines, seed, iterations=100):
    wins, loses, loses1, undecided = 0, 0, 0, 0

    for _ in range(iterations):
        result = ai_gameloop_analytical(size, num_mines, seed)
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

def simulationTorch(model, size, num_mines, seed, iterations=100):
    wins, loses, loses1, undecided = 0, 0, 0, 0

    for _ in range(iterations):
        result = ai_gameloop(model, size, num_mines, seed)
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

def load_model(input_size, hidden_size, output_size, device):
    model = MinesweeperNN(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load('minesweeper_nn.pth'))
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the device
    return model

def convert_board_to_numerical(player_board):
    # Define mapping from characters to numerical values
    mapping = {'X': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10}

    # Convert characters to numerical values based on the mapping
    numerical_board = [[mapping[cell] if cell in mapping else 0 for cell in row] for row in player_board]

    return numerical_board

def ai_take_input(model, size, player_board, device):
    row, col = -1, -1
    while(((col > 10 or col < 0) or (row > 10 or row < 0))):
        with torch.no_grad():  # Disable gradient computation during evaluation
            # Convert player_board list to tensor
            player_board_numerical = convert_board_to_numerical(player_board)
            player_board_tensor = torch.tensor(player_board_numerical, dtype=torch.float32, device=device).view(1, size * size)

            # Forward pass to get predicted move
            predicted_move = model(player_board_tensor)

            # Round the predicted move to integers
            predicted_move = torch.round(predicted_move).long()
            row = predicted_move[0][0].item()
            col = predicted_move[0][1].item()

    return row, col


def ai_gameloop(model, size, num_mines, seed, device):
    game_board, player_board = create_boards(size, num_mines, seed)
    game_started = False
    while True:
        row, col = ai_take_input(model, size, player_board, device)
        if row is None or col is None:
            # print('\nUndecided')
            return '?'

        if is_mine(game_board, row, col):
            if not game_started:
                # print('\nLose on first')
                return 'L1'
            # print('\nLose')
            return 'L'

        else:
            reveal_squares(game_board, player_board, row, col)
            game_started = True
            if is_game_finished(game_board, player_board):
                # print('\nWin')
                return 'W'


if __name__ == '__main__':
    size = 10
    num_mines = 10
    seed = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Welcome to Minesweeper!")
    print("Choose an option:")
    print("1. Play the game yourself")
    print("2. Let the AI play using Torch model")
    print("3. Let the AI play using analytical solver")

    option = input("Enter your choice (1, 2, or 3): ")

    if option == "1":
        gameloop(size, num_mines, seed)
    elif option == "2":
        input_size = size * size
        hidden_size = [200]
        output_size = 2
        model = load_model(input_size, hidden_size, output_size, device)
        iterations = 100
        wins, loses, loses1, undecided = 0, 0, 0, 0
        for _ in range(iterations):
            result = ai_gameloop(model, size, num_mines, seed, device)
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
    elif option == "3":
        simulation(size, num_mines, seed)
    else:
        print("Invalid option. Please choose 1, 2, or 3.")
