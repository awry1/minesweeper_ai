import random

def create_boards(size, num_mines, seed=None):
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

def save_board(file, board, move):
    with open(file, 'a') as f:
        for row in board:
            f.write(' '.join(row) + '\n')
        f.write(f'{move[0]} {move[1]}\n\n')

def generate_data(file, num_boards=1000, size=10, num_mines=15):
    for _ in range(num_boards):
        game_board, player_board = create_boards(size, num_mines)
        move = (random.randint(0, size-1), random.randint(0, size-1))
        save_board(file, game_board, move)

if __name__ == '__main__':
    generate_data('evaluationData.txt', num_boards=2000)
    generate_data('trainingData.txt', num_boards=10000)
