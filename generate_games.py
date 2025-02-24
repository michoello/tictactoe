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
  m_crosses = ttt.TTTClass(sys.argv[2])
  m_zeroes = ttt.TTTClass(sys.argv[3])

  g = game.Game(m_crosses, m_zeroes)
  steps, winner = g.play_game(0.0)
  for step_no, (values, board, ply, x, y, reward) in enumerate(steps):
    print("Step", step_no, ":", "crosses" if ply == 1 else "zeroes")
    game.print_scores(values)
    print("Next step:", x, y, " Reward: ", reward)
    game.print_board(board)
    print()


if mode == "play_many_games":
  m_crosses = ttt.TTTClass(sys.argv[2])
  m_zeroes = ttt.TTTClass(sys.argv[3])

  winners = {-1: 0, 0: 0, 1: 0}
  cnt = 0
  g = game.Game(m_crosses, m_zeroes)
  for f in range(100):
     #_, winner = g.play_game(0.3)
     _, winner = g.play_game(0.1)
     winners[winner] = winners[winner] + 1
     cnt = cnt + 1

  print(f"Crosses: {winners[1]} out of {cnt}")
  print(f"Zeroes: {winners[-1]} out of {cnt}")
  print(f"Ties: {winners[0]} out of {cnt}")
