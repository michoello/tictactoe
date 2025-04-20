import random
import copy
from dataclasses import dataclass
from typing import Optional

START_BOARD = [ [0 for _ in range(6)] for _ in range(6)]

@dataclass
class Step:
    step_no: int
    ply: int  # 1 for crosses, -1 for zeroes
    x: int
    y: int
    board: list[list[int]]
    values: list[list[float]]
    reward: Optional[float] = None


class Game:
   def __init__(self, model_crosses, model_zeroes):
      self.model_crosses = model_crosses
      self.model_zeroes = model_zeroes

   
   def play_game(self, exploration_rate, exploration_steps=-1):
      board = copy.deepcopy(START_BOARD)
    
      steps = []

      step_no, ply, m  = 0, 1, self.model_crosses
      winner = None
      while True:
    
        boards = all_next_steps(board, ply)
        if len(boards) == 0:
           break
    
        values = m.get_next_step_values(boards)
        if exploration_steps > 0 and step_no >= exploration_steps:
           exploration_rate = 0
        x, y = choose_next_step(values, ply, step_no, exploration_rate)
        board[x][y] = ply

        ss = Step(
            step_no=step_no,
            ply=ply,
            x=x,
            y       =y,
            board=copy.deepcopy(board),
            values=copy.deepcopy(values)
        )

        steps.append(ss)
    
        winner, _ = check_winner(board)
        if winner != 0:
           break
    
        ply = -ply
        m = self.model_crosses if ply == 1 else self.model_zeroes
        step_no = step_no + 1

      # Set desired rewards to the boards
      reward = winner
      for step in reversed(steps):
         step.reward = reward

         reward = reward * 0.9


      return steps, winner

# ----------------------------------


def generate_random_board():
    size = 6 * 6
    num_zeroes = random.randint(0, size // 2)  # Random number of zeroes (up to half the board)
    num_crosses = num_zeroes + random.choice([0, 1])  # Either equal or one more cross
    num_empty = size - num_zeroes - num_crosses
    
    values = [1] * num_crosses + [-1] * num_zeroes + [0] * num_empty
    random.shuffle(values)
    
    return [values[i * 6:(i + 1) * 6] for i in range(6)]

bgs = {
   'grey': "\033[100m",
   'black': "\033[40m" 
}

fgs = {
   'green': "\033[32m",
   'blue': "\033[94m",
   'red': "\033[31m",
}

cancel_color = "\033[0m"

def cprint(fg, bg, what):
    if bg in bgs:
       what = bgs[bg] + what + cancel_color
    if fg in fgs:
       what = fgs[fg] + what + cancel_color
    print(what, end="")
    

def print_board(board):
    winner, xyo = check_winner(board)
    
    for i, row in enumerate(board):
        for j, cell in enumerate(row):

            bg = 'grey' if (i + j) % 2 == 0 else 'black'
            if cell == -1:
               what, fg = ' O ', 'green'
            elif cell == 1:
               what, fg = ' X ', 'blue'
            else:
               what, fg = '   ', 'std'
            fg = 'red' if (i, j) in xyo else fg 
            cprint(fg, bg, what)

        print()


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
                if winner != 0 and winner != b[i][j]:
                    return None, []
                winner = b[i][j] 
                xyo = xyo + [(i, j)] + xy

    return winner, sorted(set(xyo))


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


# Generates all boards for next single step (ply=1 crosses, ply=-1 zeroes)
# Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
def all_next_steps(board, ply):
   boards = []
   for row in range(6):
       for col in range(6):
           if board[row][col] == 0:
              next_board = copy.deepcopy(board)
              next_board[row][col] = ply
              boards.append((next_board, row, col))
   return boards

# Generate sequence of boards for a single random game
def generate_random_game():
   boards = [START_BOARD]

   ply = 1 # crosses
   num = 0 # number of filled cells
   while num < 36:
     row = random.randint(0, 5)  
     col = random.randint(0, 5)

     if board[row][col] == 0:
        board[row][col] = ply
        num = num + 1
        ply = -ply

        boards.append(copy.deepcopy(board))

        winner, _ = check_winner(board)
        if winner != 0:
           return boards, winner

   return boards, 0



def print_scores(values):
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            bg_color = "\033[100m" if (i + j) % 2 == 0 else "\033[40m" 
            score = round(value * 100) if value is not None else "  "
            print(bg_color + f" {score:02}" + "\033[0m", end="")
        print()

def best_step(values, ply):
  best = -100 if ply == 1 else 100
  best_xy = (-1, -1)
  for row in range(6):
     for col in range(6):
         if (v := values[row][col]) is None:
            continue
         if ply == 1 and v > best:
             best = v
             best_xy = (row, col)
         if ply == -1 and v < best:
             best = v 
             best_xy = (row, col)
  return best_xy

# TODO: don't use values, use board here
def random_step(values, ply):
  # Reservoir sampling
  count, chosen = 0, None
  for row in range(6):
     for col in range(6):
         if values[row][col] is not None:
            count = count + 1
            if random.random() < 1 / count:
               chosen = (row, col)
  return chosen


def choose_next_step(values, ply, step_no, exploration_rate): 
    if step_no == 0 or random.random() < exploration_rate:
      x, y = random_step(values, ply)
    else:
      x, y = best_step(values, ply)
    return x, y

