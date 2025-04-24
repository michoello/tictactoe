from .ml import *


from . import ttt_classifier as tttc
from . import ttt_player as tttp

def pickup_model(tp, file):
  if tp not in ['classifier', 'player']:
      raise f'Bad type: {tp}'
  return tttc.TTTClass(file) if tp == 'classifier' else tttp.TTTPlayer(file)

