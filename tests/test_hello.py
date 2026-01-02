import unittest
from lib import ml
from lib import game
from utils import roughlyEqual


class TestHelloWorld(unittest.TestCase):

    def test_basic_ops(self):
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

    def test_bme_loss(self):

        xx = ml.BB([[1], [2]])
        yy = ml.BB([[4], [6]])

        ll = xx.mse(yy)

        self.assertEqual(ll.val(), [[9], [16]])

        ll.dif()
        self.assertEqual(ll.dval(), [[-3], [-4]])

        xx.appl(0.1)
        self.assertTrue(roughlyEqual(ll.val(), [[7.29], [12.96]]))

    # TODO: should be MCE (mean cross entropy)
    def test_mse_loss(self):
        x = ml.BB([[1, 2]])
        w = ml.BB([[3, 4, 5], [6, 7, 8]])

        y = x @ w

        self.assertEqual(y.val(), [[15, 18, 21]])

        loss = y.mse(ml.BB([[3, 4, 5]]))
        self.assertEqual(loss.val(), [[144, 196, 256]])

        loss.dif()
        self.assertEqual(loss.dval(), [[12, 14, 16]])

        # Check the gradient of weights, apply it and check that loss values decreased
        self.assertTrue(roughlyEqual(w.dval(), [[12, 14, 16], [24, 28, 32]]))
        w.appl(0.01)
        self.assertTrue(roughlyEqual(loss.val(), [[129.96, 176.89, 231.04]]))

        # Check the gradient of inputs, apply it and check that loss values decreased
        self.assertTrue(roughlyEqual(x.dval(), [[172, 298]]))
        x.appl(0.001)
        self.assertTrue(roughlyEqual(loss.val(), [[84.42, 113.07, 145.9]]))

    def test_bce_loss(self):
        x = ml.BB([[0.1, -0.2]])
        w = ml.BB([[-0.1, 0.5, 0.3], [-0.6, 0.7, 0.8]])

        y = (x @ w).sigmoid()

        self.assertTrue(roughlyEqual(y.val(), [[0.527, 0.478, 0.468]], 3))

        loss = y.bce(ml.BB([[0, 1, 0.468]]))
        self.assertTrue(roughlyEqual(loss.val(), [[0.75, 0.739, 0.691]], 3))

        # First calculate the grad for `y`, just to check values
        y.dif([[1, 1, 1]])
        self.assertTrue(roughlyEqual(y.dval(), [[0.2492, 0.2495, 0.2489]], 4))

        # Not let's check the grad starting from the `loss`, the real one
        loss.dif()
        self.assertTrue(roughlyEqual(loss.dval(), [[2.116, -2.094, -0.002]], 3))

        # Now check that `y` grads are very diffferent
        self.assertTrue(roughlyEqual(y.dval(), [[0.527, -0.522, -0.0004]], 3))
        
        self.assertTrue(roughlyEqual(w.dval(), [
             [0.0527, -0.052, -4.543/100000],
             [-0.105, 0.104, 9.086/100000]
        ], 3))

        w.appl(1.0)
        self.assertTrue(roughlyEqual(w.val(), [
            [-0.153, 0.552, 0.3],
            [-0.495, 0.596, 0.8]
        ], 3))

        # Check that loss decreased
        self.assertTrue(roughlyEqual(loss.val(), [[0.736, 0.726, 0.691]]))

        # Check that outputs are getting a bit closer
        self.assertTrue(roughlyEqual(y.val(), [[0.521, 0.484, 0.468]], 3))

        x.appl(0.01)
        self.assertTrue(roughlyEqual(x.val(), [[0.103, -0.193]], 3))

        # Check that updating x also reduces the loss
        self.assertTrue(roughlyEqual(loss.val(), [[0.734, 0.723, 0.691]], 3))

    def test_reshape(self):
        w1 = [[3, 4, 5], [6, 7, 8]]
        assert ml.reshape(w1, 3, 2) == [[3, 4], [5, 6], [7, 8]]

        ww1 = ml.BB(w1)
        self.assertEqual(ww1.val(), w1)

        ww2 = ml.BBReshape(ww1, 6, 1)
        self.assertEqual(ww2.val(), [[3], [4], [5], [6], [7], [8]])

        assert ww2.val() == [[3], [4], [5], [6], [7], [8]], f"actual value {ww2.val()}"

    def test_game_reshape(self):
        board = [
            [ 1, 0, 1, 0, 0, 0 ],
            [ 1, 0,-1, 0,-1, 0 ],
            [ 1, 0, 1, 0, 0, 0 ],
            [ 0,-1, 0,-1, 0, 0 ],
            [ 1, 0, 1, 0, 0, 0 ],
            [ 1, 0, 1, 0, 0, 0 ],
        ]

        xx = ml.BB(board)
        self.assertEqual(xx.dims(), [6, 6])

        with self.assertRaises(ValueError):
            zz0 = ml.BBReshape(xx, 1, 32)

        zz0 = ml.BBReshape(xx, 1, 36)
        self.assertEqual(zz0.dims(), [1, 36])

    def test_serialize(self):
        x = ml.BB([[1, 2, 3], [4, 5, 6]])

        x_saved = x.save()
        self.assertEqual(x_saved, "[[[1, 2, 3], [4, 5, 6]]]")

        y = ml.BB([])
        y_saved = y.save()
        self.assertEqual(y_saved, "[[]]")

        y = ml.BB([[1, 2], [3, 4], [5, 6]])
        y_saved = y.save()
        self.assertEqual(y_saved, "[[[1, 2], [3, 4], [5, 6]]]")

        z = x @ y
        z_saved = z.save()
        self.assertEqual(z.val(), [[22, 28], [49, 64]])
        self.assertEqual(
            z_saved, "[[[[1, 2, 3], [4, 5, 6]]], [[[1, 2], [3, 4], [5, 6]]]]"
        )

        x.load("[[[11, 22, 33], [44, 55, 66]]]")
        x_saved = x.save()
        self.assertEqual(x_saved, "[[[11, 22, 33], [44, 55, 66]]]")

        z_saved = z.save()
        self.assertEqual(z.val(), [[242, 308], [539, 704]])
        self.assertEqual(
            z_saved, "[[[[11, 22, 33], [44, 55, 66]]], [[[1, 2], [3, 4], [5, 6]]]]"
        )

        # Check that inner fields of the `z` are updated correctly
        self.assertEqual(z.arg(0).val(), [[11, 22, 33], [44, 55, 66]])
        self.assertEqual(z.input.val(), [[11, 22, 33], [44, 55, 66]])

    def test_game_winner(self):
        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertEqual(game.Board(board).check_winner()[0], 1)

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, -1, 1, 1, 1, 0],
            [0, -1, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertEqual(game.Board(board).check_winner()[0], -1)

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -1, 1, 1, 1, 0],
            [0, -1, 0, 0, 1, 0],
            [0, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        self.assertEqual(game.Board(board).check_winner()[0], 1)

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0],
            [0, -1, 1, -1, 1, 0],
            [0, -1, -1, 0, 1, 0],
            [0, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        winner, cells = game.Board(board).check_winner()
        self.assertEqual(winner, -1)
        self.assertEqual(cells, [(1, 4), (2, 3), (3, 2), (4, 1)])

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, -1, 0],
            [0, -1, 1, 1, 1, 0],
            [0, -1, -1, 0, 1, 0],
            [0, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        winner, cells = game.Board(board).check_winner()
        self.assertEqual(winner, None)
        self.assertEqual(cells, [])

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, -1, -1, 0, -1, 0],
            [0, -1, 1, 1, 1, 1],
            [0, -1, -1, 0, 1, 0],
            [0, -1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        winner, cells = game.Board(board).check_winner()
        self.assertEqual(winner, None)

        board = [
            [0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, -1, 0],
            [0, 1, 1, 1, 1, 1],
            [0, 1, -1, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ]
        winner, cells = game.Board(board).check_winner()
        self.assertEqual(winner, 1)
        self.assertEqual(
            cells, [(1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (3, 1), (4, 1)]
        )


if __name__ == "__main__":
    unittest.main()
