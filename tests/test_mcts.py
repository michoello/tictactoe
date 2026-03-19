import unittest
from unittest.mock import patch
from lib import game
from lib import ttt_player as ttt
from lib.mcts import best_mcts_step
from utils import SimpleRNG
from listinvert import Mod3l, Data, Convo, ReLU, Add, MatMul, SoftMax, SoftMaxCrossEntropy, Tanh, SSE, value
from tests.test_game import MyTestCase

class TestMcts(MyTestCase):
    def test_terminal_win(self) -> None:

       # Models don't matter here as we test terminal states behavior which is fixed
       model_x = ttt.TTTPlayer(enable_cpp=True)
       model_o = ttt.TTTPlayer(enable_cpp=True)
       g = game.Game(model_x, model_o, game.GameType.TICTACTOE_6_6_4, "mcts")

       # Test that MCTS chooses winning move
       board = game.Board([
           [0, 0,-1,-1, 0,-1],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1],
       ])

       # Note the step is not done by this call, it only returns coordinates
       # X to win
       game_state_x = game.GameState(board.copy(), 1)
       next_state_x = best_mcts_step(g, game_state_x, 100)
       self.assertEqual([next_state_x.x, next_state_x.y], [2, 2])
       self.assertAlmostEqualNested(next_state_x.winner, 1)
       self.assertAlmostEqualNested(next_state_x.xyo, [(2, 2), (3, 3), (4, 4), (5, 5)])

       # Put O to first row to win
       game_state_o = game.GameState(board.copy(), -1)
       next_state_o = best_mcts_step(g, game_state_o, 100)
       self.assertEqual([next_state_o.x, next_state_o.y], [0, 4])
       self.assertAlmostEqualNested(next_state_o.winner, -1)
       self.assertAlmostEqualNested(next_state_o.xyo, [ (0, 2), (0, 3), (0, 4), (0, 5)])

    def test_terminal_defense(self) -> None:

       rng = SimpleRNG(seed=2)
       with patch("random.random", new=rng.random):
           model_x = ttt.TTTPlayer(enable_cpp=True)
           model_o = ttt.TTTPlayer(enable_cpp=True)
       g = game.Game(model_x, model_o, game.GameType.TICTACTOE_6_6_4, "mcts")
       # ----------------------
       # Test that MCTS chooses protective move

       # Put X between almost winning Os in row 1:
       board = game.Board([
           [0, 0,-1,-1, 0,-1],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1],
       ])

       # Models DO matter here even with terminal states, as to discover protective
       # behavior it has to go pretty deep, and it is not always enough to have just 1500
       # simulation. Therefore using fixed random.
       #
       # So high number of simulation is due to MCTS bugs. TODO fix them and reduce
       #
       game_state_x = game.GameState(board, 1)
       next_state_x = best_mcts_step(g, game_state_x, 1500)
       self.assertEqual([next_state_x.x, next_state_x.y], [0, 4])

       # Put O in center to block Xs diagonal:
       board = game.Board([
           [0, 0,-1,-1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1],
       ])
       game_state_o = game.GameState(board, 1)
       next_state_o = best_mcts_step(g, game_state_o, 1500)
       self.assertEqual([next_state_o.x, next_state_o.y], [2, 2])

       # TODO: add tests
       # - invariants = num counts of root = sum counts of first layer, and equal to num simulations
       # - first expansion propagates value (num_counts = 1 in root, value = single child value)
       # - second simulation adds a value
       # - puct exploration/exploitation - have to manually tweak priors and mock network values
       # - test near full board (to check that tree returns early and does not crash)
       # - test tie



if __name__ == "__main__":
    unittest.main()
