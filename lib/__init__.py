from .ml import *


from . import ttt_classifier as tttc
from . import ttt_player as tttp
from . import ttt_player_v2 as tttv2
import random


def pickup_model(tp, file):
    if tp not in ["classifier", "player", "random", "playerv2"]:
        raise f"Bad type: {tp}"

    if tp == "classifier":
       return tttc.TTTClass(file)
    elif tp == "player": 
       return tttp.TTTPlayer(file)
    elif tp == "playerv2": 
       return tttv2.TTTPlayerV2(file)
    elif tp == "random":
       return tttp.TTTRandom()
    else:
       print("UNSOPPORTAD: ", tp)
       return None
