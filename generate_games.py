from lib import ml, game
import random
import sys
import copy

from lib import ttt_classifier as ttt


mode = sys.argv[1] if len(sys.argv) > 1 else "idontknow"
print(mode)


if mode == "idontknow":
   print("idontknow")
   sys.exit(0)


if mode=="random_boards":
  wins = {}
  for i in range(1000):
    board = game.generate_random_board()
    game.print_board(board)
    winner, _ = game.check_winner(board)
    print()
    wins[winner] = wins.get(winner, 0) + 1

  print("WINNERS: ", wins) 


if mode== "single_game":
   boards, winner = game.generate_random_game()

   for num, board in enumerate(boards):
       print("Number", num)
       game.print_board(board)
       print()

   print("Winner: ", winner)



if mode == "many_games":
   wins = {}
   for i in range(100):
     _, winner = game.generate_random_game()
     wins[winner] = wins.get(winner, 0) + 1

   print("WINNERS: ", wins) 


def print_scores(values):
  for row in range(6):
     for col in range(6):
         score = round(values[row][col] * 100) if values[row][col] is not None else "  "
         print(f"{score:02} ", end="")
     print("")

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

if mode == "model_game":

  with open("model_dump.json", "r") as file:
       model_dump = file.read()
       ttt.lloss.load(model_dump)


  board = copy.deepcopy(game.START_BOARD)

  ply = 1
  step_no = 0
  while True:
    print("Step ", step_no)

    boards = game.all_next_steps(board, ply)
    if len(boards) == 0:
       break

    values = copy.deepcopy(game.START_VALUES)
    for bxy in boards:
       b, x, y = bxy
       ttt.xx.set(b)
       value = ttt.zz2.val()
       values[x][y] = value[0][0]

    print_scores(values)
    x, y = best_step(values, ply)
    print("Best step:", x, y)
    board[x][y] = ply
    game.print_board(board)

    winner, _ = game.check_winner(board)
    print("Winner: ", winner)
    if winner != 0:
       break

    ply = -ply
