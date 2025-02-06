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

# Returns 1 if crosses win, -1 if zeroes win, 0 if tie and None if board is invalid
def check_winner(b):
    # Check rows and columns
    winners = {1: 0, -1: 0}

    #ll = [(0,0), (0, 1), (0, 2), (0, 3)]
    #winner = 0

    for i in range(6):
        for j in range(3):
            cell = b[i][j]
            if cell != 0 and b[i][j] == b[i][j + 1] == b[i][j + 2] == b[i][j + 3]:
                winners[ cell ] = winners[ cell ] + 1
            cell = b[j][i]
            if cell != 0 and b[j][i] == b[j + 1][i] == b[j + 2][i] == b[j + 3][i]:
                winners[ cell ] = winners[ cell ] + 1

    # Check diagonals
    for i in range(3):
        for j in range(3):
            cell = b[i][j]
            if cell != 0 and b[i][j] == b[i + 1][j + 1] == b[i + 2][j + 2] == b[i + 3][j + 3]:
                winners[ cell ] = winners[ cell ] + 1
            cell = b[i][j+3]
            if cell != 0 and b[i][j + 3] == b[i + 1][j + 2] == b[i + 2][j + 1] == b[i + 3][j]:
                winners[ cell ] = winners[ cell ] + 1

    crosses = winners[1]
    zeroes = winners[-1]

    if zeroes + crosses > 1: return None
    if crosses == 1: return 1
    if zeroes == 1:  return -1
    return 0

# Generates a random batch of size N, where each class is presented with n // 3 samples
def generate_batch(n):
  boards, winners = [], []
  for board_class in range(-1, 2):
    for i in range(n // 3):
      while True:
        board = generate_random_board()
        winner = check_winner(board)
        if winner == board_class:
          break
      boards.append(board)
      winners.append([(winner + 1.0) / 2.0])
  return boards, winners

