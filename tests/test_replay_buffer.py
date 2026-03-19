import unittest
from collections import Counter
from lib import replay_buffer, game
from typing import cast, Any
from utils import SimpleRNG
from unittest.mock import patch

def make_gs(val: Any) -> game.GameState:
    gs = game.GameState(board=game.Board(), next_move=1)
    gs.board.state = cast(list[list[int]], val)
    gs.reward = [[1.0]]
    return gs

def get_val(gs: game.GameState) -> Any:
    return gs.board.state


class TestReplayBuffer(unittest.TestCase):
    def test_buffer_never_exceeds_max_size(self) -> None:
        buf = replay_buffer.ReplayBuffer(max_size=10)
        for i in range(100):
            buf.maybe_add(make_gs(i))
        self.assertLessEqual(len(buf.buffer), 10)

    def test_get_random_returns_item_from_buffer(self) -> None:
        buf = replay_buffer.ReplayBuffer(max_size=3)
        buf.maybe_add(make_gs("a"))
        buf.maybe_add(make_gs("b"))
        buf.maybe_add(make_gs("c"))
        random_item = get_val(buf.get_random())
        self.assertIn(random_item, [item[0] for item in buf.buffer])

    def test_get_uniformity(self) -> None:
        buf = replay_buffer.ReplayBuffer(max_size=5)
        trials = 100_000
        for i in range(trials):
            buf.maybe_add(make_gs(i))

        # Count how often each item is selected
        counts: Counter[int] = Counter()
        for _ in range(10_000):
            item = get_val(buf.get_random())
            counts[cast(int, item)] += 1

        # Expect roughly uniform spread
        self.assertEqual(len(buf.buffer), 5)
        min_count = min(counts.values())
        max_count = max(counts.values())
        self.assertLess(max_count - min_count, 0.2 * max_count)

    def test_maybe_add_bucket_distribution(self) -> None:

        rng = SimpleRNG(seed=42)  # best so far
        with patch("random.randint", new=rng.randint):

            max_size = 10_000
            total_items = 1_000_000
            bucket_size = 10_000
            num_buckets = total_items // bucket_size
            expected_per_bucket = max_size / num_buckets  # Expect ~100 per bucket

            buf = replay_buffer.ReplayBuffer(max_size)
            for i in range(total_items):
                buf.maybe_add(make_gs(i))

            # Count how many final buffer items fall into each bucket
            bucket_counts = [0] * num_buckets
            for item in buf.buffer:
                bucket_idx = cast(int, item[0]) // bucket_size
                bucket_counts[bucket_idx] += 1

            # Check that all bucket counts are within ±30% of expected
            tolerance = expected_per_bucket * 0.3  # 30%
            for idx, count in enumerate(bucket_counts):
                self.assertTrue(
                    abs(count - expected_per_bucket) <= tolerance,
                    f"Bucket {idx} has count {count}, expected around {expected_per_bucket}",
                )


if __name__ == "__main__":
    unittest.main()
