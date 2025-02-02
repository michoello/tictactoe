import unittest
from lib import say_hello
from lib import ml


def roughlyEqual(m1, m2):
    if all([round(a, 2) == round(b, 2) for row_a, row_b in zip(m1, m2) for a, b in zip(row_a, row_b)]):
        return True
    print(f"Arrays not equal:\n{m1}\n{m2}")
    return False


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


        assert yy.val() == [
            [ 15, 18, 21 ],
        ]


        ww_wrong = ml.BB(w_wrong)
        with self.assertRaises(ValueError):
           y_wrong = xx @ ww_wrong

        b = [[1, 2, 3]]

        yy1 = yy + ml.BB(b)


        assert yy1.val() == [[16, 20, 24]], f"actual value is {yy1.val()}"


    def test_bme_loss(self):
        
        xx = ml.BB([[1], [2]])
        yy = ml.BB([[4], [6]])

        ll = xx.mse(yy)

        self.assertEqual(ll.val(), [[9], [16]])

        ll.dif()
        self.assertEqual(ll.dval(), [[-3], [-4]])

        xx.appl(0.1)
        self.assertTrue(roughlyEqual(ll.val(), [[7.29], [12.96]]))


    def test_bme_loss(self):
        x = ml.BB([[1, 2]])
        w = ml.BB([[3, 4, 5], [ 6, 7, 8]])

        y = x @ w

        self.assertEqual(y.val(), [[15, 18, 21]])

        loss = y.mse(ml.BB([[0, 0, 0]]))
        self.assertEqual(loss.val(), [[225, 324, 441]])

        loss.dif()
        self.assertEqual(loss.dval(), [[15, 18, 21]])

        w.appl(0.01)
        self.assertTrue(roughlyEqual(loss.val(), [[ 203.06, 292.41, 398.00]]))

        x.appl(0.01)
        self.assertTrue(roughlyEqual(loss.val(), [[ 195.02, 284.87, 391.68 ]]))



    def test_reshape(self):
        w1 = [[3, 4, 5], [6, 7, 8]] 
        assert ml.reshape(w1, 3, 2) == [ [3, 4], [5, 6], [7, 8]]

        ww1 = ml.BB(w1)
        self.assertEqual(ww1.val(), w1)

        ww2 = ml.BBReshape(ww1, 6, 1)
        self.assertEqual(ww2.val(), [[3],[4],[5],[6],[7],[8]])

        assert ww2.val() == [[3],[4],[5],[6],[7],[8]], f"actual value {ww2.val()}"





if __name__ == "__main__":
    unittest.main()
