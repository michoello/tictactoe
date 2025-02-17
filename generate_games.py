from lib import game
import sys
import copy
import random

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

if mode == "model_game":

  m_crosses = ttt.TTTClass()
  m_zeroes = ttt.TTTClass()

  m_zeroes.load_from_file("models/model_trained.json")
  m_crosses.load_from_file("models/model_initial.json")

  board = copy.deepcopy(game.START_BOARD)

  ply = 1
  m = m_crosses
  step_no = 0
  while True:
    print("Step", step_no, ":", "crosses" if ply == 1 else "zeroes")

    boards = game.all_next_steps(board, ply)
    if len(boards) == 0:
       break

    values = copy.deepcopy(game.START_VALUES)
    for bxy in boards:
       b, x, y = bxy
       m.x.set(b)
       value = m.prediction.val()
       values[x][y] = value[0][0]

    game.print_scores(values)

    if step_no > 0 and random.random() < 0.9:
      x, y = game.best_step(values, ply)
      print("Best step:", x, y)
    else:
      x, y = game.random_step(values, ply)
      print("Random step:", x, y)

    board[x][y] = ply
    game.print_board(board)

    winner, _ = game.check_winner(board)
    if winner != 0:
       print("Winner: ", winner)
       break

    print()
    ply = -ply
    m = m_crosses if ply == 1 else m_zeroes
    step_no = step_no + 1
