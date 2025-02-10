from lib import game
import random
import sys

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




if mode == "idontknow":
   print("idontknow")
