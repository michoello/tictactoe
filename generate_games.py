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


if mode == "generate_random_game":
   boards, winner = game.generate_random_game()

   for num, board in enumerate(boards):
       print("Step", num)
       game.print_board(board)
       print()

   print("Winner: ", winner)



if mode == "many_games":
   wins = {}
   for i in range(100):
     _, winner = game.generate_random_game()
     wins[winner] = wins.get(winner, 0) + 1

   print("WINNERS: ", wins) 

   

if mode == "play_single_game":

  m_crosses = ttt.TTTClass()
  m_zeroes = ttt.TTTClass()

  crosses_model = sys.argv[2]
  zeroes_model = sys.argv[3]

  #m_zeroes.load_from_file("models/model_trained.json")
  #m_crosses.load_from_file("models/model_initial.json")
  m_crosses.load_from_file(crosses_model)
  m_zeroes.load_from_file(zeroes_model)

  g = game.Game(m_crosses, m_zeroes)
  steps = g.play_game(0.3)
  for step_no, (values, board, ply, x, y, reward) in enumerate(steps):
    print("Step", step_no, ":", "crosses" if ply == 1 else "zeroes")
    game.print_scores(values)
    print("Next step:", x, y, " Reward: ", reward)
    game.print_board(board)
    print()
