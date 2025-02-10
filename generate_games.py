from lib import ml, game
import random
import sys

from lib import ttt_classifier as ttt


mode = sys.argv[1] if len(sys.argv) > 1 else "idontknow"
print(mode)

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


  with open("model_dump.json", "r") as file:
       model_dump = file.read()
       ttt.lloss.load(model_dump)


  ttt.xx.set([
     [1, -1, 1, 1, 1,-1], 
     [1, -1,-1, 1,-1,-1],
     [-1, 1,-1,-1,-1,-1],
     [1,  1,-1,-1, 1, 1],
     [1, -1, 1,-1, 1, 1],
     [1, -1, 1, 0, 1,-1]
  ])

  check_prediction2 = ttt.zz2.val()
  print("Loading: check_prediction=", check_prediction2)



if mode == "idontknow":
   print("idontknow")
