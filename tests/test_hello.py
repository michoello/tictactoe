import unittest
from helloworld import say_hello
from helloworld import ml

class TestHelloWorld(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(say_hello("Alice"), "Hello, Alice!")
        self.assertEqual(say_hello("Bob"), "Hello, Bob!")
        self.assertNotEqual(say_hello("Charlie"), "Hello, Alice!")


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


    def test_reshape(self):
        w1 = [[3, 4, 5], [6, 7, 8]] 
        assert ml.reshape(w1, 3, 2) == [ [3, 4], [5, 6], [7, 8]]

        ww1 = ml.BB(w1)

        ww2 = ml.BBReshape(ww1, 6, 1)

        assert ww2.val() == [[3],[4],[5],[6],[7],[8]], f"actual value {ww2.val()}"





if __name__ == "__main__":
    unittest.main()
