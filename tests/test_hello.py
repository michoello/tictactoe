import unittest
from helloworld import say_hello

class TestHelloWorld(unittest.TestCase):
    def test_say_hello(self):
        self.assertEqual(say_hello("Alice"), "Hello, Alice!")
        self.assertEqual(say_hello("Bob"), "Hello, Bob!")
        self.assertNotEqual(say_hello("Charlie"), "Hello, Alice!")

if __name__ == "__main__":
    unittest.main()
