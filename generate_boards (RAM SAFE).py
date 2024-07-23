import os

# Constants for quick change
SIZE = 5
NUM_MINES = 10


def generate_all_boards(size, num_mines, filename):
    # Assume square board
    if size * size < num_mines:
        raise ValueError('Number of mines cannot exceed number of cells')
    
    if size > 5:
        raise ValueError('You don\'t want to wait for eternity')
    
    num_boards = 0
    MAX = (2**(num_mines) - 1) << (size * size - num_mines)
    PERCENTAGE = MAX // 100
    with open(filename, 'a') as file:  # Open file once
        print(f'Iterations: {MAX + 1}')
        for num in range(MAX, -1, -1):
            if num % PERCENTAGE == 0:
                print(f'{100 - num // PERCENTAGE}% done')

            if bin(num).count('1') <= num_mines:
                board = [[' ' for _ in range(size)] for _ in range(size)]
                for i in range(size):
                    for j in range(size):
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
    for i in range(0, NUM_MINES+1):
        num_to_expect += Newton(SIZE * SIZE, i)
    print(f'Number of boards to expect: {num_to_expect}')

    os.makedirs('GENERATED_BOARDS', exist_ok=True)
    FILENAME = os.path.join('GENERATED_BOARDS', f'Boards_{SIZE}.txt')
    num_boards = generate_all_boards(SIZE, NUM_MINES, FILENAME)

    print(f'Number of boards generated: {num_boards}')
