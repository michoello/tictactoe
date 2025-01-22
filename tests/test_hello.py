import unittest
from helloworld import say_hello
from helloworld import ml

class TestHelloWorld(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(say_hello("Alice"), "Hello, Alice!")
        self.assertEqual(say_hello("Bob"), "Hello, Bob!")
        self.assertNotEqual(say_hello("Charlie"), "Hello, Alice!")


    def test_ml(self):
        x = [[0.5, 0.9]] 
        w1 = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] 
        
        xx = ml.BB(x)
        ww1 = ml.BB(w1)

        yy = xx @ ww1

        assert xx.dims() == [1, 2]
        assert ww1.dims() == [3, 2], f"Actual value is {ww1.dims()}"
        assert yy.dims() == [1, 2], f"Actual value is {yy.dims()}"



        print(x)
        print(w1)
        print(ml.transpose(w1))
        print(ml.matmul(x, w1))

        #return [[sum(a * b for a, b in zip(row, col)) for col in transpose(B)] for row in A]
        print( [[ (row, col) for col in ml.transpose(w1)] for row in x] )


if __name__ == "__main__":
    unittest.main()
