import random
import time
from typing import Callable
from listinvert import Matrix, multiply_matrix

def benchmark(func: Callable[[], None]) -> float:
    """Measure runtime of a function in seconds (float)."""
    start = time.perf_counter()
    func()
    end = time.perf_counter()
    return end - start


# Equivalent to C++ benchmark_multiply
def benchmark_multiply() -> None:
    A = Matrix(100, 100)
    B = Matrix(100, 100)
    A.fill_uniform()
    B.fill_uniform()

    C = Matrix(100, 100)
    for i in range(5000):
        multiply_matrix(A, B, C)
        if i % 100 == 0:
            print(i)

    print("Benchmark completed ✅")


if __name__ == "__main__":
    elapsed = benchmark(benchmark_multiply)
    print(f"Elapsed time: {elapsed:.3f} seconds")
