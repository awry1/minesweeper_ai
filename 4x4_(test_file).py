# Not up to date
import os
import torch
import random
from sympy import *
from numpy import *

from minesweeperAI import *

# Constants for quick change
SIZE = 12
DEFAULT_MINES = 10
RAND_MINES = False
SEED = None

# ANALYTICAL = True
# TORCH = False

ITERATIONS_ANALYTICAL = 1000
# ITERATIONS_TORCH = 1000

# MODEL_NAME = 'NN'
# TRAIN_NAME = 'DATA'
# TEST_NAME = 'TEST'

# MODEL_DIRECTORY = 'torchModels'
# TRAIN_DIRECTORY = 'trainData'
# TEST_DIRECTORY = 'testResults'

def test_fun(size, default_mines, rand_mines, seed):
    if seed is not None:
        random.seed(seed)

    wins, loses, loses1, undecided = 0, 0, 0, 0
    for _ in range(ITERATIONS_ANALYTICAL):
        if _ % 100 == 0:
            print(f'Progress: {_}/{ITERATIONS_ANALYTICAL}')
        result = test(size, default_mines, rand_mines)
        if result == '?':
            undecided += 1
        elif result == 'W':
            wins += 1
        elif result == 'L1':
            loses1 += 1
        else:
            loses += 1

    print(f'\nWins: {wins}, Loses: {loses}, Loses on first: {loses1}, Undecided: {undecided}')


def test(size, default_mines, rand_mines):
    num_mines = random_num_mines(default_mines, rand_mines)
    game_board, player_board = create_boards(size, num_mines)

    filename = 'board_data.txt'
    sub_size = 4

    last_input = None
    game_started = False
    while True:
        # print_board(player_board)

        row, col, risk_board = take_input_analytical1(size, num_mines, player_board, game_started)

        if last_input == (row, col):
            save_board_data_to_file(size, sub_size, player_board, risk_board, filename)
            return '?'

        if not game_started:
            game_board = ensure_fair_start(size, num_mines, game_board, row, col)

        last_input = row, col

        if is_mine(game_board, row, col):
            # print_end_board(game_board, player_board)
            if not game_started:
                return 'L1'
            save_board_data_to_file(size, sub_size, player_board, risk_board, filename)
            return 'L'

        else:
            game_started = True

            reveal_squares(game_board, player_board, row, col)
            if is_game_finished(game_board, player_board):
                # print_board(player_board)
                save_board_data_to_file(size, sub_size, player_board, risk_board, filename)
                return 'W'

def take_input_analytical1(size, num_mines, player_board, game_started):
    moves = []
    risk_board = [[1.0 for _ in range(size)] for _ in range(size)]
    sub_size = 4
    filename = 'board_data.txt'

    if not game_started:
        row, col = random.randint(0, size - 1), random.randint(0, size - 1)
    else:
        moves, mines = solve_analytical1(size, player_board)
        risk_board = update_risk_board(num_mines, player_board, risk_board, moves, mines)

        if len(moves) > 0:
            row, col = moves[0]
        else:
            row, col = choose_least_risky_move(risk_board)

    return row, col, risk_board

def save_board_data_to_file(size, sub_size, player_board, risk_board, filename):
    with open(filename, 'a') as file:
        for i in range(0, size, sub_size):
            for j in range(0, size, sub_size):
                sub_board = [player_board[i + x][j:j + sub_size] for x in range(sub_size) if i + x < size]
                risk_sub_board = [risk_board[i + x][j:j + sub_size] for x in range(sub_size) if i + x < size]

                # Check if the sub_board is not fully discovered and contains at least one known cell
                if any(cell == '?' for row in sub_board for cell in row) and \
                        any(cell != '?' for row in sub_board for cell in row):
                    # Ensure that unknown cells are marked as '?'
                    sub_board_display = [['?' if cell == ' ' else cell for cell in row] for row in sub_board]

                    for row in sub_board_display:
                        file.write(' '.join(row) + '\n')
                    file.write('Risk factors:\n')
                    for row in risk_sub_board:
                        file.write(' '.join(f'{risk:.2f}' for risk in row) + '\n')
                    file.write('\n')

def divide_board(size, sub_size):
    sub_boards = []
    for i in range(0, size, sub_size):
        for j in range(0, size, sub_size):
            sub_board = [(i+x, j+y) for x in range(sub_size) for y in range(sub_size) if i+x < size and j+y < size]
            sub_boards.append(sub_board)
    return sub_boards

def solve_analytical1(size, player_board):
    sub_size = 4
    sub_boards = divide_board(size, sub_size)
    all_moves = []
    all_mines = []

    for sub_board in sub_boards:
        undiscovered = find_undiscovered_fields_subboard(sub_board, player_board)
        if not undiscovered:
            continue

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

        all_moves.extend(moves)
        all_mines.extend(mines)

    return all_moves, all_mines

def find_undiscovered_fields_subboard(sub_board, player_board):
    undiscovered = []
    for pos in sub_board:
        if player_board[pos[0]][pos[1]] == ' ':
            neighbours = []
            for i in range(pos[0] - 1, pos[0] + 2):
                for j in range(pos[1] - 1, pos[1] + 2):
                    if 0 <= i < len(player_board) and 0 <= j < len(player_board[0]) and player_board[i][j].isdigit():
                        neighbours.append((i, j))
            if neighbours:
                undiscovered.append((pos, neighbours))
    return undiscovered


########################## Main starts here ##########################
if __name__ == '__main__':
    test_fun(SIZE, DEFAULT_MINES, RAND_MINES, SEED)