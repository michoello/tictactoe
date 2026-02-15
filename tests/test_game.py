import unittest
from lib import ml
from lib import game
import tempfile
from lib import ttt_player as ttt
from lib import ttt_player_v2 as tttv2
from utils import roughlyEqual
from utils import SimpleRNG
from unittest.mock import patch
from lib import ratings

# TODO: rename value
from listinvert import (
    Matrix,
    multiply_matrix,
    Mod3l,
    Block,
    Data,
    MatMul,
    SSE,
    Abs,
    Add,
    BCE,
    Sigmoid,
    Reshape,
    value,
    Convo,
    ReLU,
    SoftMax,
    SoftMaxCrossEntropy,
    Tanh,
)


def DData(mod3l, rows, cols, values):
    res = Data(mod3l, rows, cols)
    mod3l.set_data(res, values)
    return res


class MyTestCase(unittest.TestCase):
    def assertAlmostEqualNested(self, a, b, delta=1e-3):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            self.assertEqual(len(a), len(b), "Lengths differ")
            for x, y in zip(a, b):
                self.assertAlmostEqualNested(x, y, delta)
        else:
            self.assertAlmostEqual(a, b, delta=delta)

class TestTrainingCycle(MyTestCase):

    def test_omg(self):
        x = [[1, 2]]
        w1 = [[3, 4, 5], [6, 7, 8]]
        w_wrong = [[3, 4, 5]]

        xx = ml.BB(x)
        ww1 = ml.BB(w1)

        yy = xx @ ww1

        assert xx.dims() == [1, 2]
        assert ww1.dims() == [2, 3], f"Actual value is {ww1.dims()}"
        assert yy.dims() == [1, 3], f"Actual value is {yy.dims()}"

        assert yy.val() == [[15, 18, 21]]

        ww_wrong = ml.BB(w_wrong)
        with self.assertRaises(ValueError):
            y_wrong = xx @ ww_wrong

        b = [[1, 2, 3]]

        yy1 = yy + ml.BB(b)

        assert yy1.val() == [[16, 20, 24]], f"actual value is {yy1.val()}"

        # update x values
        xx.set([[11, 22]])
        self.assertEqual(xx.val(), [[11, 22]])
        # Check that yy (which is xx @ ww) is also updated
        self.assertEqual(yy.val(), [[165, 198, 231]])

    def test_training_player_and_game_cpp(self):
        rng = SimpleRNG(seed=45)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            init_model = tempfile.mktemp()
            trained_model = tempfile.mktemp()

            m = ttt.TTTPlayer(enable_cpp=True)
            m.save_to_file(init_model)
            m.save_to_file(trained_model)

            random_model = ttt.TTTPlayer(init_model, enable_cpp=True)
            m = ttt.TTTPlayer(trained_model, enable_cpp=True)

            print("Training")

            g = game.Game(random_model, random_model)
            test_boards, test_values = g.generate_batch_from_games(20)

            total_epochs = 50 
            for epoch in range(total_epochs):
                if epoch % 2 == 0:
                  g = game.Game(m, random_model)
                else:
                  g = game.Game(random_model, m)
                  
                train_boards, train_values = g.generate_batch_from_games(20)

                for i in range(10):
                    for board, val in zip(train_boards, train_values):
                        m.set_board_and_value( 1, board, val)
                        m.set_board_and_value(-1, board, val)
                        m.calc_grads()
                        m.apply_gradient()

                    test_loss = 0
                    for board, val in zip(test_boards, test_values):
                        m.set_board_and_value( 1, board, val)
                        m.set_board_and_value(-1, board, val)
                        test_loss = test_loss + value(m.model_x.loss.fval())[0][0]

                if epoch % 5 == 0:
                    m.save_to_file(trained_model)
                    print(f"{epoch/total_epochs*100}% - test_loss {test_loss}")


            print("Playing...")
            trained_model = ttt.TTTPlayer(trained_model, enable_cpp=True)

            # ctw = crosses_trained_winners
            g = game.Game(trained_model, random_model)
            ctw = g.competition(20)
            print("Trained crosses WINNERS cross:", ctw[1], " zero:", ctw[-1])
            self.assertGreater(ctw[1], ctw[-1])

            # ztw = zeroes_trained_winners
            g = game.Game(random_model, trained_model)
            ztw = g.competition(20)
            print("Trained zeroes WINNERS cross:", ztw[1], " zero:", ztw[-1])
            self.assertLess(ztw[1], ztw[-1])



    def test_mod3l_sse_with_grads(self):
        m = Mod3l()

        dy = Data(m, 1, 2)
        m.set_data(dy, [[1, 2]])  # true labels

        # "labels"
        dl = Data(m, 1, 2)
        m.set_data(dl, [[0, 4]])

        ds = SSE(dy, dl)


        self.assertEqual(value(ds.fval()), [[5]])

        # Derivative of loss function is its value is 1.0 (aka df/df)
        self.assertEqual(
            value(ds.bval()),
            [
                [0],
            ],
        )
        # Derivative of its args
        self.assertEqual(
            value(dy.bval()),
            [
                [2, -4],
            ],
        )

        dy.apply_bval(0.1)
        self.assertAlmostEqualNested(
            value(dy.fval()),
            [
                [0.8, 2.4],
            ],
        )

        # Calc loss again
        self.assertAlmostEqualNested(
            value(ds.fval()),
            [
                [3.2],
            ],
        )

    def test_training_player_single_iter(self):
        rng = SimpleRNG(seed=1)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            mx = ttt.TTTPlayer(enable_cpp=True)
            mo = ttt.TTTPlayer(enable_cpp=True)

            g = game.Game(mx, mo)
            train_boards, train_values = g.generate_batch_from_games(25)

            mx.set_board_and_value( 1, train_boards[0], train_values[0])
            mx.set_board_and_value(-1, train_boards[0], train_values[0])

            # Check that gradient decreased
            self.assertAlmostEqualNested(value(mx.model_x.loss.fval()), [[1.647714]], 1e-6)
            mx.calc_grads()
            mx.apply_gradient()
            self.assertAlmostEqualNested(value(mx.model_x.loss.fval()), [[1.631000]], 1e-6)


    def test_py_cpp_models_compare(self):
        rng = SimpleRNG(seed=1)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            m_py = ttt.TTTPlayer()

        rng = SimpleRNG(seed=1)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            m_cpp = ttt.TTTPlayer(enable_cpp=True)

        self.assertAlmostEqualNested(m_py.model_x.x.val(), value(m_cpp.model_x.x.fval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.z1.val(), value(m_cpp.model_x.z1.fval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.z3.val(), value(m_cpp.model_x.z3.fval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.loss.val(), value(m_cpp.model_x.loss.fval()), 1e-6)

        m_py.calc_grads()
        m_cpp.calc_grads()

        # The "flow" or "operational" blocks have this discrepancy a bit between
        # old python and new cpp implementations:
        self.assertAlmostEqualNested(m_py.model_x.loss.dval(), value(m_cpp.model_x.z3.bval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.z3.dval(), value(m_cpp.model_x.za.bval()), 1e-6)

        # But the weights back values (grads) are consistent:
        self.assertAlmostEqualNested(m_py.model_x.w3.dval(), value(m_cpp.model_x.w3.bval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.b3.dval(), value(m_cpp.model_x.b3.bval()), 1e-6)

        self.assertAlmostEqualNested(m_py.model_x.w2.dval(), value(m_cpp.model_x.w2.bval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.b2.dval(), value(m_cpp.model_x.b2.bval()), 1e-6)

        self.assertAlmostEqualNested(m_py.model_x.w1.dval(), value(m_cpp.model_x.w1.bval()), 1e-6)
        self.assertAlmostEqualNested(m_py.model_x.b1.dval(), value(m_cpp.model_x.b1.bval()), 1e-6)

        # Now apply grads and check that thee results match
        m_py.apply_gradient()
        m_cpp.apply_gradient()

        self.assertAlmostEqualNested(m_py.model_x.loss.val(), value(m_cpp.model_x.loss.fval()), 1e-6)

        m_py_file = tempfile.mktemp()
        m_py.save_to_file(m_py_file)

        # After saving and reload the model loss is almost the same (note 1e-3)
        m_py2 = ttt.TTTPlayer(m_py_file)
        self.assertAlmostEqualNested(m_py.model_x.loss.val(), m_py2.model_x.loss.val(), 1e-3)

        m_cpp_file = tempfile.mktemp()
        m_cpp.save_to_file(m_cpp_file)

        # Note: no enable_cpp flag -> it restores everything based on file content
        m_cpp2 = ttt.TTTPlayer(m_cpp_file) 

        # Since cpp model does not store the inputs, we copy them from first cpp model
        m_cpp2.model_x.m.set_data(m_cpp2.model_x.x, value(m_cpp.model_x.x.fval()))
        m_cpp2.model_x.m.set_data(m_cpp2.model_x.y, value(m_cpp.model_x.y.fval()))
        self.assertAlmostEqualNested(value(m_cpp.model_x.loss.fval()), value(m_cpp2.model_x.loss.fval()), 1e-3)

        # The outputs are now a bit different between py and cpp, because of rounding
        # (both weights and the inputs)
        # TODO: rounding
        self.assertAlmostEqualNested(value(m_cpp2.model_x.loss.fval()), m_py2.model_x.loss.val(), 1e-3)


    def test_ratings(self):
        rats = ratings.elo_ratings([ ('a', 1, 'b', 0) ])
        self.assertAlmostEqual(rats['a'], 1510)
        self.assertAlmostEqual(rats['b'], 1490)

        rats = ratings.elo_ratings([ ('a', 50, 'b', 50) ])
        self.assertAlmostEqual(rats['a'], 1500, delta=1)
        self.assertAlmostEqual(rats['b'], 1500, delta=1)


class TestMcts(MyTestCase):
    def test_terminal_win(self):

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
       row, col = g.best_mcts_step(game_state_x, 100)
       self.assertAlmostEqual([row, col], [2, 2])
       self.assertAlmostEqualNested(game_state_x.winner, 1)
       self.assertAlmostEqualNested(game_state_x.xyo, [(2, 2), (3, 3), (4, 4), (5, 5)])

       # Put O to first row to win
       game_state_o = game.GameState(board.copy(), -1)
       row, col = g.best_mcts_step(game_state_o, 100)
       self.assertAlmostEqual([row, col], [0, 4])
       self.assertAlmostEqualNested(game_state_o.winner, -1)
       self.assertAlmostEqualNested(game_state_o.xyo, [ (0, 2), (0, 3), (0, 4), (0, 5)])

    def test_terminal_defense(self):

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
       row, col = g.best_mcts_step(game_state_x, 1500)
       self.assertEqual([row, col], [0, 4])

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
       row, col = g.best_mcts_step(game_state_o, 1500)
       self.assertEqual([row, col], [2, 2])

       # TODO: add tests
       # - invariants = num counts of root = sum counts of first layer, and equal to num simulations
       # - first expansion propagates value (num_counts = 1 in root, value = single child value)
       # - second simulation adds a value
       # - puct exploration/exploitation - have to manually tweak priors and mock network values
       # - test near full board (to check that tree returns early and does not crash)
       # - test tie

    # Clone of cpp testcase larger_model
    def test_larger_model(self):
        m = Mod3l()
        dinput = Data(m, 3, 3)
        m.set_data(
            dinput,
            [
                [1, 0, -1],
                [1, 0, -1],
                [0, 1, -1],
            ],
        )

        dkernel1 = Data(m, 2, 2)
        m.set_data(
            dkernel1,
            [
                [0.3, 0.1],
                [0.2, 0.0],
            ],
        )
        dc1 = Convo(dinput, dkernel1)
        rl1 = ReLU(dc1)

        dkernel2 = Data(m, 2, 2)
        m.set_data(
            dkernel2,
            [
                [-0.3, 0.1],
                [-0.2, 0.4],
            ],
        )
        dc2 = Convo(dinput, dkernel2)
        rl2 = ReLU(dc2)

        rl = Add(rl1, rl2)

        dw = Data(m, 3, 3)
        m.set_data(dw, [[1, 2, 3], [5, 6, 7], [9, 10, 11]])

        dlogits = MatMul(rl, dw)
        dsoftmax = SoftMax(dlogits)

        # This is our toy policy network head
        dlabels = Data(m, 3, 3)
        m.set_data(
            dlabels,
            [
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
        )
        policy_loss = SoftMaxCrossEntropy(dlogits, dsoftmax, dlabels)

        dw2 = Data(m, 3, 1)
        m.set_data(dw2, [[1.5], [2.5], [3.5]])

        dvalue = Tanh(MatMul(rl, dw2))

        dlabel = Data(m, 1, 1)
        m.set_data(dlabel, [[-1]])

        dvalue_loss = SSE(dvalue, dlabel)

        self.assertAlmostEqualNested(value(dvalue_loss.fval()), [[3.998]])
        self.assertAlmostEqualNested(value(policy_loss.fval()), [[2.302]])

        for i in range(10):
            value_before = dvalue_loss.fval().get(0, 0)
            policy_before = policy_loss.fval().get(0, 0)

            dkernel1.apply_bval(0.01)
            dkernel2.apply_bval(0.01)
            dw.apply_bval(0.01)
            dw2.apply_bval(0.01)

            value_after = dvalue_loss.fval().get(0, 0)
            policy_after = policy_loss.fval().get(0, 0)

            assert (
                value_before > value_after
            ), f"Value loss did not decrease. Before:{value_before}, after:{value_after}"
            assert (
                policy_before > policy_after
            ), f"Policy lose did not decrease. Before:{value_before}, after:{value_after}"

        self.assertAlmostEqualNested(value(dvalue_loss.fval()), [[3.995]])
        self.assertAlmostEqualNested(value(policy_loss.fval()), [[1.593]])

class TestPlayerV2(MyTestCase):

    # UNDER CONSTRUCTION.
    # Add needed methods, update needed methods, init with rands, and test 
    # actual game and training
    #
    def test_debugging(self):
      rng = SimpleRNG(seed=45)
      with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
      ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
      ):
        player2 = tttv2.TTTPlayerV2()
        player2.set_board_and_value(
            player = 1,
            board = [
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0,-1],
                [1, 0, 0, 0, 0,-1],
                [0, 0, 0, 0, 0,-1],
                [1, 0, 0, 0, 0,-1],
                [1, 0, 0, 0, 0, 0],
            ],
            _value = [[1]],
            policy = [
                [0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0.6, 0] +
                [0, 0, 0, 0, 0, 0] +
                [0.3, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0,0.1]
              ]
            
        )
 
        value_loss, policy_loss = player2.get_loss_value()
        self.assertAlmostEqualNested(value_loss, 3.322)
        self.assertAlmostEqualNested(policy_loss, 6.146)

        for i in range(1200):
            value_before, policy_before = player2.get_loss_value()
            player2.apply_gradient(0.001)
            value_after, policy_after = player2.get_loss_value()

            assert (
                  value_before > value_after
            ), f"Value loss did not decrease. Before:{value_before}, after:{value_after}, iteration {i}"
            assert (
                  policy_before > policy_after
            ), f"Policy lose did not decrease. Before:{policy_before}, after:{policy_after}, iteration {i}"


        value_loss, policy_loss = player2.get_loss_value()
        self.assertAlmostEqualNested(value_loss, 0.00248)
        self.assertAlmostEqualNested(policy_loss, 0.902)

        # TODO: get rid of "impl"
        self.assertAlmostEqualNested(value(player2.impl.policy.fval()), [
                [0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0.6, 0] +
                [0, 0, 0, 0, 0, 0] +
                [0.3, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0, 0] +
                [0, 0, 0, 0, 0,0.1]
              ], delta=0.01)


    def test_training_player_and_game_v2(self):
        rng = SimpleRNG(seed=44)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            init_model = tempfile.mktemp()
            trained_model = tempfile.mktemp()

            m = tttv2.TTTPlayerV2()
            m.save_to_file(init_model)
            m.save_to_file(trained_model)

            random_model = tttv2.TTTPlayerV2(init_model)
            m = tttv2.TTTPlayerV2(trained_model)

            print("Training")

            g = game.Game(random_model, random_model)

            #test_boards, test_values = g.generate_batch_from_games(20)

            total_epochs = 50
            for epoch in range(total_epochs):
                #if epoch % 2 == 0:
                g = game.Game(m, random_model)
                #else:
                #  g = game.Game(random_model, m)
                  

                train_boards, train_values = g.generate_batch_from_games(20)
                test_boards, test_values = g.generate_batch_from_games(20)

                for i in range(4):
                    train_loss = 0
                    for board, val in zip(train_boards, train_values):
                        m.set_board_and_value( 1, board, val)
                        m.apply_gradient(0.001)
                        train_loss = train_loss + m.get_loss_value()[0]

                    test_loss = 0
                    for board, val in zip(test_boards, test_values):
                        m.set_board_and_value( 1, board, val)
                        test_loss = test_loss + m.get_loss_value()[0]

                if epoch % 10 == 0:
                    m.save_to_file(trained_model)
                    print(f"{epoch/total_epochs*100}% - test_loss {test_loss}")

            print("Playing...")
            trained_model = tttv2.TTTPlayerV2(trained_model)

            # ctw = crosses_trained_winners
            g = game.Game(trained_model, random_model)
            ctw = g.competition(20)
            print("Trained crosses WINNERS cross:", ctw[1], " zero:", ctw[-1])
            self.assertGreater(ctw[1], ctw[-1])

            # ztw = zeroes_trained_winners
            g = game.Game(random_model, trained_model)
            ztw = g.competition(20)
            print("Trained zeroes WINNERS cross:", ztw[1], " zero:", ztw[-1])
            # TODO: uncomment. There is a lot to do to make it work yet ahead
            #self.assertLess(ztw[1], ztw[-1])




if __name__ == "__main__":
    unittest.main()
