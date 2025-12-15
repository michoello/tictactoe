import unittest
from unittest.mock import patch

import string
import random
from utils import SimpleRNG
from lib import ml
from utils import roughlyEqual
from utils import compress, decompress


class TestSimpleRNG(unittest.TestCase):
    def test_random_reproducibility(self):
        rng1 = SimpleRNG(12345)
        seq1 = [rng1.random() for _ in range(100)]

        rng2 = SimpleRNG(12345)
        seq2 = [rng2.random() for _ in range(100)]

        # elements are the same both times
        self.assertEqual(seq1, seq2)
        # all elements are unique
        self.assertEqual(len(seq1), len(set(seq1)))

    def test_repro_random_matrix(self):
        rng = SimpleRNG(seed=42)
        with patch("random.random", new=rng.random):
            m = ml.random_matrix(6, 6)
            self.assertTrue(
                roughlyEqual(
                    m,
                    [
                        [-0.060, 0.617, 0.963, 0.471, 0.940, 0.994],
                        [0.770, -0.637, 0.374, 0.104, -0.974, 0.776],
                        [0.452, 0.600, 0.476, 0.669, 0.603, 0.194],
                        [0.480, -0.959, 0.776, -0.3, -0.465, -0.765],
                        [0.289, -0.003, -0.664, -0.416, 0.737, -0.414],
                        [0.359, -0.827, -0.033, 0.133, -0.171, 0.151],
                    ],
                    2,
                )
            )

    def test_compress_decompress(self):
        s = 'hello'
        self.assertEqual(decompress(compress(s)), s)

        s = 'asdcjhalskjdh lqercAW #fcasdcj;lkjh ca!32v cadk bjbasdc #$adc'
        self.assertEqual(decompress(compress(s)), s)
        
        s = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(1000))
        self.assertEqual(decompress(compress(s)), s)

if __name__ == "__main__":
    unittest.main()
