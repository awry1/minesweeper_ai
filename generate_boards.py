# Currently not used and not compatible with current version
import os

# Constants for quick change
SIZE = 5, 5     # X, Y
NUM_MINES = 4


def generate_all_boards(size, num_mines, filename):
    size_x, size_y = size
    if size_x * size_y < num_mines:
        raise ValueError('Number of mines cannot exceed number of cells')
    
    if size_x * size_y > 25:
        raise ValueError('You don\'t want to wait for eternity')
    
    num_boards = 0
    MAX = (2**(num_mines) - 1) << (size_x * size_y - num_mines)
    PERCENTAGE = MAX // 100
    with open(filename, 'a') as file:  # Open file once
        print(f'Iterations: {MAX + 1}')
        for num in range(MAX, -1, -1):
            if num % PERCENTAGE == 0:
                print(f'{100 - num // PERCENTAGE}% done')

            if bin(num).count('1') <= num_mines:
                board = [[' ' for _ in range(size_x)] for _ in range(size_y)]
                for i in range(size_y):
                    for j in range(size_x):
                        if num & 1:
                            board[i][j] = 'X'
                        num >>= 1
                num_boards += 1

                # Write board to file
                for row in board:
                    row_str = ' '.join(['_' if cell == ' ' else cell for cell in row])
                    file.write(row_str + '\n')
                file.write('\n')

    return num_boards


def Newton(n, k):
    if k == 0:
        return 1
    return Newton(n, k-1) * (n - k + 1) // k


if __name__ == '__main__':
    num_to_expect = 0
    size_x, size_y = SIZE
    for i in range(0, NUM_MINES+1):
        num_to_expect += Newton(size_x * size_y, i)
    print(f'Number of boards to expect: {num_to_expect}')

    os.makedirs('GENERATED_BOARDS', exist_ok=True)
    FILENAME = os.path.join('GENERATED_BOARDS', f'Boards_{SIZE}.txt')
    num_boards = generate_all_boards(SIZE, NUM_MINES, FILENAME)

    print(f'Number of boards generated: {num_boards}')
