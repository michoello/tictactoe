from lib import game
import sys
import copy
import random

from typing import Dict
#from lib import ttt_classifier as tttc
#from lib import ttt_player as tttp
from lib import tttc, tttp, pickup_model


import argparse

parser = argparse.ArgumentParser(description="Play your model")

parser.add_argument('--mode', type=str, help='how to run this script')
parser.add_argument('--crosses_model', type=str, help='Type and path of crosses model')
parser.add_argument('--zeroes_model', type=str, help='Type and path of zeroes model')
args = parser.parse_args()


if args.mode == "idontknow":
   print("idontknow")
   sys.exit(0)


if args.mode=="random_boards":
  wins: Dict[int, int] = {}
  for i in range(1000):
    board = game.generate_random_board()
    board.print_board()
    winner, _ = game.check_winner(board)
    print()
    wins[winner] = wins.get(winner, 0) + 1

  print("WINNERS: ", wins) 


if args.mode == "generate_random_game":
   boards, winner = game.generate_random_game()

   for num, board in enumerate(boards):
       print("Step", num)
       board.print_board()
       print()

   print("Winner: ", winner)



if args.mode == "many_games":
   wins = {}
   for i in range(100):
     _, winner = game.generate_random_game()
     wins[winner] = wins.get(winner, 0) + 1

   print("WINNERS: ", wins) 


if args.mode == "play_single_game":
  m_crosses = pickup_model(*args.crosses_model.split(':'))
  m_zeroes = pickup_model(*args.zeroes_model.split(':'))

  g = game.Game(m_crosses, m_zeroes)
  steps, winner = g.play_game(0.5, 2)
  for ss in steps:
    print("Step", ss.step_no, ":", "crosses" if ss.ply == 1 else "zeroes")
    game.print_scores(ss.values)
    print("  Move:", ss.x, ss.y, " Reward: ", ss.reward)
    ss.board.print_board()
    print()




if args.mode == "play_many_games":
  m_crosses = pickup_model(*args.crosses_model.split(':'))
  m_zeroes = pickup_model(*args.zeroes_model.split(':'))

  num_games = 100
  winners = game.competition(m_crosses, m_zeroes, num_games)

  print(f"Crosses: {winners[1]} out of {num_games}")
  print(f"Zeroes: {winners[-1]} out of {num_games}")
  print(f"Ties: {winners[0]} out of {num_games}")
