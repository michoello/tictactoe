import random

def generate_random_board():
    size = 6 * 6
    num_zeroes = random.randint(0, size // 2)  # Random number of zeroes (up to half the board)
    num_crosses = num_zeroes + random.choice([0, 1])  # Either equal or one more cross
    num_empty = size - num_zeroes - num_crosses
    
    values = [1] * num_crosses + [-1] * num_zeroes + [0] * num_empty
    random.shuffle(values)
    
    return [values[i * 6:(i + 1) * 6] for i in range(6)]


def print_board(board):
    symbols = {
        0: "   ", 
        -1: "\033[32m 0 \033[0m",  # Green for zero
        1: "\033[94m X \033[0m"   # Light blue for cross
    }
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            bg_color = "\033[100m" if (i + j) % 2 == 0 else "\033[40m" 
            print(bg_color + symbols[cell] + "\033[0m", end="")
        print()


def check_winner(b):
    # Check rows and columns
    for i in range(6):
        for j in range(3): 
            if b[i][j] == b[i][j + 1] == b[i][j + 2] == b[i][j + 3] and b[i][j] != 0:
                return b[i][j]
            if b[j][i] == b[j + 1][i] == b[j + 2][i] == b[j + 3][i] and b[j][i] != 0:
                return b[j][i]

    # Check diagonals
    for i in range(3):
        for j in range(3):
            if b[i][j] == b[i + 1][j + 1] == b[i + 2][j + 2] == b[i + 3][j + 3] and b[i][j] != 0:
                return b[i][j]
            if b[i][j + 3] == b[i + 1][j + 2] == b[i + 2][j + 1] == b[i + 3][j] and b[i][j + 3] != 0:
                return b[i][j + 3]

    return 0 
