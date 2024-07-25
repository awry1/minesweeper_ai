# Very much a work in progress
# Below some scribbles which may or may not be useful
from solve_analytical import *


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


def save_game_state_5x5(window, risk, filename):
    with open(filename, 'a') as file:
        for row in window:
            row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
            file.write(row_str + '\n')

        file.write('Risk:\n')
        file.write(f'{risk}\n')

        file.write('\n')


if __name__ == '__main__':
    print('Very much in progress')
    print('File not meant to be run')