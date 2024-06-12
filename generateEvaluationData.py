import os

from minesweeperAI import create_boards, is_game_finished, ai_take_input, reveal_squares, is_mine


def generate_evaluation_data(size, num_mines, num_boards):
    evaluation_data = []

    for _ in range(num_boards):
        game_board, player_board = create_boards(size, num_mines, seed=None)
        game_started = False

        moves = []

        while not is_game_finished(game_board, player_board):
            row, col = ai_take_input(size, game_started, player_board=player_board)
            if (row is None and col is None) or is_mine(game_board, row, col):
                game_board, player_board = create_boards(size, num_mines, seed=None)
                game_started = False
                continue  # Skip the game if no move made
            if game_started:
                moves.append((row, col))
                break
            reveal_squares(game_board, player_board, row, col)
            game_started = True

        if moves:  # If any moves have been made, add the game to the evaluation data
            board_state = player_board.copy()  # Create a deep copy of the player board
            move = moves[0]
            evaluation_data.append((board_state, move))

    return evaluation_data

if __name__ == '__main__':
    size = 10
    num_mines = 10
    num_boards = 100  # Number of evaluation boards to generate
    filename = 'evaluationData.txt'

    evaluation_data = generate_evaluation_data(size, num_mines, num_boards)

    if os.path.exists(filename):
        # Remove the file if it exists
        os.remove(filename)

    # Save the evaluation data to a text file
    with open(filename, 'w') as file:
        for board_state, move in evaluation_data:
            for row in board_state:
                row_str = ' '.join(['?' if cell == ' ' else cell for cell in row])
                file.write(row_str + '\n')
            file.write(f"{move[0]} {move[1]}\n")