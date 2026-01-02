from lib import game
import sys
import copy
import random

from typing import Dict

from lib import tttc, tttp, pickup_model


import argparse

parser = argparse.ArgumentParser(description="Play your model")

parser.add_argument("--mode", type=str, help="how to run this script")
parser.add_argument("--model_x", type=str, help="Type and path of crosses model")
parser.add_argument("--model_o", type=str, help="Type and path of zeroes model")
parser.add_argument("--game_type", type=str, help="Game type: 4 or 5_tor")
parser.add_argument("--num_games", type=int, default=100, help="how many games to play")
parser.add_argument("--game_mode", type=str, default="greedy", help="Game mode: greedy, minimax, mcts")
args = parser.parse_args()


if args.mode == "idontknow":
    print("idontknow")
    sys.exit(0)


game_type = args.game_type
game_type = game.GameType.TICTACTOE_6_6_5_TOR if game_type == "5_tor" else game.GameType.TICTACTOE_6_6_4

game_mode = args.game_mode
model_x = pickup_model(*args.model_x.split(":"))
model_o = pickup_model(*args.model_o.split(":"))

if args.mode == "play_single_game":
    g = game.Game(model_x, model_o, game_type, game_mode)
    steps = g.play_game()
    for ss in steps:
        ss.print_state()


if args.mode == "play_many_games":
    g = game.Game(model_x, model_o, game_type, game_mode)
    winners = g.competition(args.num_games)
    print(
        f"Crosses: {winners[1]}, Zeroes: {winners[-1]}, Ties: {winners[0]} out of {args.num_games}"
    )
