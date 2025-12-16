import unittest
from lib import ml
from lib import game
import tempfile
from lib import ttt_player as ttt
from utils import roughlyEqual
from utils import SimpleRNG
from unittest.mock import patch
from lib import ratings

from listinvert import invert, Matrix, multiply_matrix, Mod3l, Block, Data, MatMul, SSE, Reshape, Sigmoid, Add, BCE

def DData(mod3l, rows, cols, values):
    res = Data(mod3l, rows, cols)
    mod3l.set_data(res, values)
    return res


class TestTrainingCycle(unittest.TestCase):
    def assertAlmostEqualNested(self, a, b, delta=1e-3):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            self.assertEqual(len(a), len(b), "Lengths differ")
            for x, y in zip(a, b):
                self.assertAlmostEqualNested(x, y, delta)
        else:
            self.assertAlmostEqual(a, b, delta=delta)

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

    def test_training_player_and_game_py(self):
        #return
        rng = SimpleRNG(seed=45)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            init_model = tempfile.mktemp()
            trained_model = tempfile.mktemp()

            m = ttt.TTTPlayer()
            m.save_to_file(init_model)

            print("Training")
            total_epochs = 25  # can be as little as 10, but let's keep it as this
            test_boards, test_winners = game.generate_batch(20)
            for epoch in range(total_epochs):
                train_boards, train_winners = game.generate_batch(20)

                for i in range(2):
                    for board, winner in zip(train_boards, train_winners):
                        m.x.set(board)
                        m.y.set([winner])
                        m.calc_grads()
                        m.apply_gradient()

                    test_loss = 0
                    for board, winner in zip(test_boards, test_winners):
                        m.x.set(board)
                        m.y.set([winner])
                        test_loss = test_loss + m.loss.val()[0][0]

                if epoch % 5 == 0:
                    m.save_to_file(trained_model)
                    print(f"{epoch/total_epochs*100}% - test_loss {test_loss}")


            print("Playing...")
            random_model = ttt.TTTPlayer(init_model)
            trained_model = ttt.TTTPlayer(trained_model)

            # ctw = crosses_trained_winners
            ctw = game.competition(trained_model, random_model, 20)
            print("Trained crosses WINNERS cross:", ctw[1], " zero:", ctw[-1])
            self.assertGreater(ctw[1], ctw[-1])

            # ztw = zeroes_trained_winners
            ztw = game.competition(random_model, trained_model, 20)
            print("Trained zeroes WINNERS cross:", ztw[1], " zero:", ztw[-1])
            self.assertLess(ztw[1], ztw[-1])


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

            print("Training")
            total_epochs = 25  # can be as little as 10, but let's keep it as this
            test_boards, test_winners = game.generate_batch(20)
            for epoch in range(total_epochs):
                train_boards, train_winners = game.generate_batch(20)

                for i in range(2):
                    for board, winner in zip(train_boards, train_winners):
                        m.m.set_data(m.x, board)
                        m.m.set_data(m.y, [winner])
                        m.loss.calc_fval()
                        m.calc_grads()
                        m.apply_gradient()

                    test_loss = 0
                    for board, winner in zip(test_boards, test_winners):
                        m.m.set_data(m.x, board)
                        m.m.set_data(m.y, [winner])

                        m.loss.calc_fval()
                        test_loss = test_loss + m.loss.fval()[0][0]

                if epoch % 5 == 0:
                    m.save_to_file(trained_model)
                    print(f"{epoch/total_epochs*100}% - test_loss {test_loss}")


            print("Playing...")
            random_model = ttt.TTTPlayer(init_model, enable_cpp=True)
            trained_model = ttt.TTTPlayer(trained_model, enable_cpp=True)

            # ctw = crosses_trained_winners
            ctw = game.competition(trained_model, random_model, 20)
            print("Trained crosses WINNERS cross:", ctw[1], " zero:", ctw[-1])
            self.assertGreater(ctw[1], ctw[-1])

            # ztw = zeroes_trained_winners
            ztw = game.competition(random_model, trained_model, 20)
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

        ds.calc_fval()

        self.assertEqual(ds.fval(), [[5]])

        # Calc derivatives
        dy.calc_bval()

        # Derivative of loss function is its value is 1.0 (aka df/df)
        self.assertEqual(
            ds.bval(),
            [
                [1],
            ],
        )
        # Derivative of its args
        self.assertEqual(
            dy.bval(),
            [
                [2, -4],
            ],
        )

        dy.apply_bval(0.1)
        self.assertAlmostEqualNested(
            dy.fval(),
            [
                [0.8, 2.4],
            ],
        )

        # Calc loss again
        ds.calc_fval()
        self.assertAlmostEqualNested(
            ds.fval(),
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
            m = ttt.TTTPlayer()

            train_boards, train_winners = game.generate_batch(3)
            m.x.set(train_boards[0])
            m.y.set(train_winners)

            self.assertAlmostEqualNested(m.loss.val(), [[0.005717]], 1e-6)
            m.calc_grads()
            m.apply_gradient()
            self.assertAlmostEqualNested(m.loss.val(), [[0.005709]], 1e-6)


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
            m_cpp.loss.calc_fval()

        self.assertAlmostEqualNested(m_py.x.val(), m_cpp.x.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.z1.val(), m_cpp.z1.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.z3.val(), m_cpp.z3.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.loss.val(), m_cpp.loss.fval(), 1e-6)

        m_py.calc_grads()
        m_cpp.calc_grads()

        # The "flow" or "operational" blocks have this discrepancy a bit between
        # old python and new cpp implementations:
        self.assertAlmostEqualNested(m_py.loss.dval(), m_cpp.z3.bval(), 1e-6)
        self.assertAlmostEqualNested(m_py.z3.dval(), m_cpp.za.bval(), 1e-6)

        # But the weights back values (grads) are consistent:
        self.assertAlmostEqualNested(m_py.w3.dval(), m_cpp.w3.bval(), 1e-6)
        self.assertAlmostEqualNested(m_py.b3.dval(), m_cpp.b3.bval(), 1e-6)

        self.assertAlmostEqualNested(m_py.w2.dval(), m_cpp.w2.bval(), 1e-6)
        self.assertAlmostEqualNested(m_py.b2.dval(), m_cpp.b2.bval(), 1e-6)

        self.assertAlmostEqualNested(m_py.w1.dval(), m_cpp.w1.bval(), 1e-6)
        self.assertAlmostEqualNested(m_py.b1.dval(), m_cpp.b1.bval(), 1e-6)

        # Now apply grads and check that thee results match
        m_py.apply_gradient()
        m_cpp.apply_gradient()

        m_cpp.loss.calc_fval()
        self.assertAlmostEqualNested(m_py.loss.val(), m_cpp.loss.fval(), 1e-6)

        m_py_file = tempfile.mktemp()
        m_py.save_to_file(m_py_file)

        # After saving and reload the model loss is almost the same (note 1e-3)
        m_py2 = ttt.TTTPlayer(m_py_file)
        self.assertAlmostEqualNested(m_py.loss.val(), m_py2.loss.val(), 1e-3)

        m_cpp_file = tempfile.mktemp()
        m_cpp.save_to_file(m_cpp_file)

        # Note: no enable_cpp flag -> it restores everything based on file content
        m_cpp2 = ttt.TTTPlayer(m_cpp_file) 

        # Since cpp model does not store the inputs, we copy them from first cpp model
        m_cpp2.m.set_data(m_cpp2.x, m_cpp.x.fval())
        m_cpp2.m.set_data(m_cpp2.y, m_cpp.y.fval())
        m_cpp2.loss.calc_fval()
        self.assertAlmostEqualNested(m_cpp.loss.fval(), m_cpp2.loss.fval(), 1e-3)

        # The outputs are now a bit different between py and cpp, because of rounding
        # (both weights and the inputs)
        # TODO: rounding
        self.assertAlmostEqualNested(m_cpp2.loss.fval(), m_py2.loss.val(), 1e-3)


    def test_ratings(self):
         rats = ratings.elo_ratings([ ('a', 1, 'b', 0) ])
         self.assertAlmostEqual(rats['a'], 1510)
         self.assertAlmostEqual(rats['b'], 1490)

         rats = ratings.elo_ratings([ ('a', 50, 'b', 50) ])
         self.assertAlmostEqual(rats['a'], 1500, delta=1)
         self.assertAlmostEqual(rats['b'], 1500, delta=1)


if __name__ == "__main__":
    unittest.main()
