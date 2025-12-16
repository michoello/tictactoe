from lib import game
import sys
import copy
import random

from typing import Dict

# from lib import ttt_classifier as tttc
# from lib import ttt_player as tttp
from lib import tttc, tttp, pickup_model


import argparse

parser = argparse.ArgumentParser(description="Play your model")

parser.add_argument("--mode", type=str, help="how to run this script")
parser.add_argument("--model_x", type=str, help="Type and path of crosses model")
parser.add_argument("--model_o", type=str, help="Type and path of zeroes model")
parser.add_argument("--game_type", type=str, help="Game type: 4 or 5_tor")
parser.add_argument("--num_games", type=int, default=100, help="how many games to play")
parser.add_argument("--game_mode", type=str, default="greedy", help="Game mode: greedy or minimax")
args = parser.parse_args()


if args.mode == "idontknow":
    print("idontknow")
    sys.exit(0)


if args.mode == "random_boards":
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

game_type = args.game_type
game_type = game.GameType.TICTACTOE_6_6_5_TOR if game_type == "5_tor" else game.GameType.TICTACTOE_6_6_4

game_mode = args.game_mode


if args.mode == "play_single_game":
    model_x = pickup_model(*args.model_x.split(":"))
    model_o = pickup_model(*args.model_o.split(":"))
    g = game.Game(model_x, model_o, game_type, game_mode)
    steps, winner = g.play_game()
    step_no = 0
    for ss in steps:
        print("Step", step_no, ":", "crosses" if ss.ply == 1 else "zeroes")
        print("  Move:", ss.x, ss.y, " Reward: ", ss.reward)
        ss.board.print_board()
        print()
        step_no += 1


if args.mode == "play_many_games":
    model_x = pickup_model(*args.model_x.split(":"))
    model_o = pickup_model(*args.model_o.split(":"))
    num_games = args.num_games
    winners = game.competition(model_x, model_o, num_games, game_type)

    print(
        f"Crosses: {winners[1]}, Zeroes: {winners[-1]}, Ties: {winners[0]} out of {num_games}"
    )
