import math
from typing import Iterable, List, Sequence, TypeVar, Optional

T = TypeVar("T")


class SimpleRNG:
    """
    Small reproducible PRNG implemented from scratch.
    - Seeding uses splitmix64 to initialize internal state.
    - Core generator: xorshift128+ variant (64-bit arithmetic).
    Methods:
      - seed(s)
      - rand_uint64() -> 64-bit unsigned int
      - random() -> float in [0, 1)
      - randint(a, b) inclusive
      - randrange(start, stop=None, step=1)
      - choice(seq)
      - shuffle(x) in-place
      - sample(population, k)
      - gauss(mu=0, sigma=1)
    """

    def __init__(self, seed: int = 0):
        # internal state: two 64-bit unsigned ints
        self._s0 = 0
        self._s1 = 0
        self._have_gauss = False
        self._gauss_value = 0.0
        self.seed(seed)

    @staticmethod
    def _mask64(x: int) -> int:
        return x & ((1 << 64) - 1)

    @staticmethod
    def _rotl(x: int, r: int) -> int:
        r %= 64
        return SimpleRNG._mask64((x << r) | (x >> (64 - r)))

    @staticmethod
    def _splitmix64(state: int) -> int:
        # splitmix64 produces 64-bit values from a 64-bit state
        z = SimpleRNG._mask64(state + 0x9E3779B97F4A7C15)
        z = SimpleRNG._mask64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = SimpleRNG._mask64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return z ^ (z >> 31)

    def seed(self, seed: int) -> None:
        # Normalize seed to 64-bit integer (allow negative seeds)
        seed = int(seed) & ((1 << 64) - 1)
        # Use splitmix64 to expand the single seed into two non-zero 64-bit state words.
        # Avoid the state being both zero.
        s = seed
        self._s0 = self._splitmix64(s)
        s = self._mask64(s + 1)
        self._s1 = self._splitmix64(s)
        # If both zero (very unlikely), set a fixed fallback
        if self._s0 == 0 and self._s1 == 0:
            self._s0 = 0x0123456789ABCDEF
            self._s1 = 0xFEDCBA9876543210
        self._have_gauss = False
        self._gauss_value = 0.0

    def rand_uint64(self) -> int:
        # xorshift128+ variant
        s1 = self._s0
        s0 = self._s1
        result = self._mask64(s0 + s1)

        s1 = self._mask64(s1 ^ (s1 << 23))
        s1 = self._mask64(s1 ^ (s1 >> 17))
        s1 = self._mask64(s1 ^ s0)
        s1 = self._mask64(s1 ^ (s0 >> 26))

        self._s0 = s0
        self._s1 = s1
        return result

    def random(self) -> float:
        # produce float in [0,1) using 53 bits of precision like CPython's random.random()
        # pull 64 bits and shift to get 53 top bits
        u = self.rand_uint64()
        # take top 53 bits
        top53 = u >> (64 - 53)
        return top53 / (1 << 53)

    def randint(self, a: int, b: int) -> int:
        if b < a:
            raise ValueError("randint: b must be >= a")
        # inclusive randint: map uniform float to integer range
        return a + self.randrange(0, b - a + 1)

    def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        if stop is None:
            # single-argument form: [0, start)
            lo, hi = 0, start
        else:
            lo, hi = start, stop
        if step == 0:
            raise ValueError("step argument must not be zero")
        if (hi - lo) <= 0 and (hi - lo) % step == 0:
            return lo  # empty or single value edge-case
        # compute number of possible values
        width = hi - lo
        if step != 1:
            # compute count carefully
            if (width > 0 and step > 0) or (width < 0 and step < 0):
                n = (abs(width) + abs(step) - 1) // abs(step)
            else:
                n = 0
        else:
            n = max(0, width)
        if n <= 0:
            raise ValueError("empty range for randrange()")
        # choose uniform index in [0, n)
        # We'll use rejection sampling to avoid bias
        while True:
            # get a 64-bit integer and reduce modulo n with rejection sampling
            r = self.rand_uint64()
            if n & (n - 1) == 0:
                # power of two, fast path
                idx = r & (n - 1)
                return lo + idx * step
            else:
                # rejection sampling: only accept r if within floor(2^64 / n) * n
                bound = (1 << 64) - ((1 << 64) % n)
                if r < bound:
                    idx = r % n
                    return lo + idx * step

    def choice(self, seq: Sequence[T]) -> T:
        if len(seq) == 0:
            raise IndexError("choice() from empty sequence")
        idx = self.randrange(len(seq))
        return seq[idx]

    def shuffle(self, x: List[T]) -> None:
        # in-place Fisher-Yates shuffle, deterministic
        for i in range(len(x) - 1, 0, -1):
            j = self.randrange(i + 1)
            x[i], x[j] = x[j], x[i]

    def sample(self, population: Sequence[T], k: int) -> List[T]:
        if k < 0 or k > len(population):
            raise ValueError("sample() larger than population or negative")
        # If k is small relative to n, one can use reservoir or set; here we'll use simple approach:
        pool = list(population)
        self.shuffle(pool)
        return pool[:k]

    def gauss(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        # Box-Muller transform using stored second value for efficiency
        if self._have_gauss:
            self._have_gauss = False
            return mu + sigma * self._gauss_value
        # generate two uniform numbers in (0,1]
        # we use random() but ensure it is not zero
        while True:
            u1 = self.random()
            if u1 > 0.0:
                break
        u2 = self.random()
        r = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        z0 = r * math.cos(theta)
        z1 = r * math.sin(theta)
        self._have_gauss = True
        self._gauss_value = z1
        return mu + sigma * z0
