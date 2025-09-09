from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Tuple

def run_parallel(tasks: List[Tuple[Callable, List[Any]]], max_workers: int) -> List[Any]:
    """
    Run multiple tasks in parallel using processes and return all results (blocking).
    
    Args:
        tasks: List of tuples (function, args_list)
            - function: callable
            - args_list: list of positional arguments for the function
        max_workers: number of processes to run in parallel
    
    Returns:
        List of results in the same order as tasks
    """
    results = [None] * len(tasks)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(func, *args): i
            for i, (func, args) in enumerate(tasks)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
    
    return results

import time


if __name__ == "__main__":

  def add(a, b):
    print("add ", a, b)
    time.sleep(1)
    print("added ", a, b)
    return a + b

  def mul(a, b):
    print("mul ", a, b)
    time.sleep(2)
    print("muled ", a, b)
    return a * b

  tasks = [
    (add, [1, 2]),
    (mul, [3, 4]),
    (add, [10, 20]),
  ]

  results = run_parallel(tasks, max_workers=2)
  print(results)  # [3, 12, 30]
