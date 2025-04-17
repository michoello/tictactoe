from lib import game
import sys
import copy
import random

from lib import ttt_classifier as tttc
from lib import ttt_player as tttp

import argparse

parser = argparse.ArgumentParser(description="Play your model")

parser.add_argument('--mode', type=str, help='how to run this script')
parser.add_argument('--crosses_model', type=str, help='Type and path of crosses model')
parser.add_argument('--zeroes_model', type=str, help='Type and path of zeroes model')
args = parser.parse_args()


#args.mode = sys.argv[1] if len(sys.argv) > 1 else "idontknow"
print(args.mode)


if args.mode == "idontknow":
   print("idontknow")
   sys.exit(0)


if args.mode=="random_boards":
  wins = {}
  for i in range(1000):
    board = game.generate_random_board()
    game.print_board(board)
    winner, _ = game.check_winner(board)
    print()
    wins[winner] = wins.get(winner, 0) + 1

  print("WINNERS: ", wins) 


if args.mode == "generate_random_game":
   boards, winner = game.generate_random_game()

   for num, board in enumerate(boards):
       print("Step", num)
       game.print_board(board)
       print()

   print("Winner: ", winner)



if args.mode == "many_games":
   wins = {}
   for i in range(100):
     _, winner = game.generate_random_game()
     wins[winner] = wins.get(winner, 0) + 1

   print("WINNERS: ", wins) 

   


def pickup_model(tp, file):
  if tp not in ['classifier', 'player']:
      raise f'Bad type: {tp}'
  return tttc.TTTClass(file) if tp == 'classifier' else tttp.TTTPlayer(file)


if args.mode == "play_single_game":
  crosses_type, crosses_file = args.crosses_model.split(':')
  m_crosses = pickup_model(crosses_type, crosses_file)
  zeroes_type, zeroes_file = args.zeroes_model.split(':')
  m_zeroes = pickup_model(zeroes_type, zeroes_file) 

  g = game.Game(m_crosses, m_zeroes)
  steps, winner = g.play_game(0.5, 2)
  for step_no, (values, board, ply, x, y, reward) in enumerate(steps):
    print("Step", step_no, ":", "crosses" if ply == 1 else "zeroes")
    game.print_scores(values)
    print("Next step:", x, y, " Reward: ", reward)
    game.print_board(board)
    print()




if args.mode == "play_many_games":
  crosses_type, crosses_file = args.crosses_model.split(':')
  m_crosses = pickup_model(crosses_type, crosses_file)
  zeroes_type, zeroes_file = args.zeroes_model.split(':')
  m_zeroes = pickup_model(zeroes_type, zeroes_file) 

  winners = {-1: 0, 0: 0, 1: 0}
  cnt = 0
  g = game.Game(m_crosses, m_zeroes)
  for f in range(100):
     #_, winner = g.play_game(0.3)
     _, winner = g.play_game(0.5, 2)
     winners[winner] = winners[winner] + 1
     cnt = cnt + 1

  print(f"Crosses: {winners[1]} out of {cnt}")
  print(f"Zeroes: {winners[-1]} out of {cnt}")
  print(f"Ties: {winners[0]} out of {cnt}")
