from lib import game


wins = {}
for i in range(1000):
   board = game.generate_random_board()
   game.print_board(board)
   winner = game.check_winner(board)
   print("Winner:", winner)
   wins[winner] = wins.get(winner, 0) + 1



print("WINNERS: ", wins) 
