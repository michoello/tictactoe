import unittest
from lib import ml
from lib import game
import tempfile
from lib import ttt_player as ttt
from utils import roughlyEqual
from utils import SimpleRNG
from unittest.mock import patch

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

    def test_training_player_and_game(self):
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
                        m.loss.dif()
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
            m.loss.dif()
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
#            m = Mod3l()
#            x = DData(m, 6, 6, ml.random_matrix(6, 6)) 
#            w1 = DData(m, 36, 64, ml.random_matrix(36, 64))
#            b1 = DData(m, 1, 64, ml.random_matrix(1, 64))
#
#            w2 = DData(m, 64, 32, ml.random_matrix(64, 32))
#            b2 = DData(m, 1, 32, ml.random_matrix(1, 32))
#
#            w3 = DData(m, 32, 1, ml.random_matrix(32, 1))
#            b3 = DData(m, 1, 1, ml.random_matrix(1, 1))
#
#            z0 = Reshape(x, 1, 36)
#            z1 = Sigmoid(Add(MatMul(z0, w1), b1))
#            z2 = Sigmoid(Add(MatMul(z1, w2), b2))
#            #z3 = Sigmoid(Add(MatMul(z2, w3), b3))
#            zm = MatMul(z2, w3)
#            za = Add(zm, b3)
#            z3 = Sigmoid(za)
#
#            y = DData(m, 1, 1, ml.random_matrix(1, 1))
#            loss = BCE(z3, y)

            m_cpp.loss.calc_fval()

        self.assertAlmostEqualNested(m_py.x.val(), m_cpp.x.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.z1.val(), m_cpp.z1.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.z3.val(), m_cpp.z3.fval(), 1e-6)
        self.assertAlmostEqualNested(m_py.loss.val(), m_cpp.loss.fval(), 1e-6)

        m_py.loss.dif()

        m_cpp.w1.calc_bval()
        m_cpp.w2.calc_bval()
        m_cpp.w3.calc_bval()
        m_cpp.b1.calc_bval()
        m_cpp.b2.calc_bval()
        m_cpp.b3.calc_bval()

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

#        w1.apply_bval(0.01)
#        b1.apply_bval(0.01)
#        w2.apply_bval(0.01)
#        b2.apply_bval(0.01)
#        w3.apply_bval(0.01)
#        b3.apply_bval(0.01)
#
        m_cpp.loss.calc_fval()
        self.assertAlmostEqualNested(m_py.loss.val(), m_cpp.loss.fval(), 1e-6)

if __name__ == "__main__":
    unittest.main()
