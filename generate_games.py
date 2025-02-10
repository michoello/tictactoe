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
 
   board = [
     [ 0, 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0],
   ]
 
   ply = 1 # crosses
   num = 0 # number of filled cells
   while True:
     row = random.randint(0, 5)  
     col = random.randint(0, 5)

     if board[row][col] == 0:
        board[row][col] = ply
        num = num + 1
        ply = -ply

        print("Number", num)
        game.print_board(board)

        winner, _ = game.check_winner(board)
        if winner != 0:
           print("Winner: ", winner)
           break
 
        print()
    
if mode == "idontknow":
   print("idontknow")
