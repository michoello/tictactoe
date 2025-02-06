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
        1: "\033[94m X \033[0m",   # Light blue for cross

        -2: "\033[31m 0 \033[0m",  # Red for winning zero
        2: "\033[31m X \033[0m",  # Red for winning cross
    }

    winner, xyo = check_winner(board)
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if (i, j) in xyo:
               cell = cell * 2
            bg_color = "\033[100m" if (i + j) % 2 == 0 else "\033[40m" 
            print(bg_color + symbols[cell] + "\033[0m", end="")
        print()
    print("Winner:", winner)

# Returns 1 if crosses win, -1 if zeroes win, 0 if tie and None if board is invalid
def check_winner(b):

    lll = [
       [(0, 1), (0, 2), (0, 3)],
       [(1, 0), (2, 0), (3, 0)],
       [(1, 1), (2, 2), (3, 3)],
       [(-1, 1), (-2, 2), (-3, 3)],
    ]

    g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

    xyo = []
    winner = 0
    for i in range(6):
       for j in range(6):
          if b[i][j] == 0:
             continue

          for ll in lll:
             xy = [(i + lx, j + ly) for lx, ly in ll]
             if all([ g(x,y) == b[i][j] for x, y in xy]):
                if winner:
                    return None, []
                winner = b[i][j] 
                xyo = xy + [(i, j)]

    return winner, xyo


# Generates a random batch of size N, where each class is presented with n // 3 samples
def generate_batch(n):
  boards, winners = [], []
  for board_class in range(-1, 2):
    for i in range(n // 3):
      while True:
        board = generate_random_board()
        winner, _ = check_winner(board)
        if winner == board_class:
          break
      boards.append(board)
      winners.append([(winner + 1.0) / 2.0])
  return boards, winners

