import unittest
import os
import shutil
import tempfile
from unittest.mock import patch
from utils import SimpleRNG
from lib import train_loop

class TestTrainLoop(unittest.TestCase):
    def test_main_loop_creates_models(self) -> None:
        # Create a temporary directory to store the models
        test_dir = tempfile.mkdtemp()
        prefix = os.path.join(test_dir, "model")

        try:
            # Use deterministic random
            rng = SimpleRNG(seed=42)
            with patch("random.random", new=rng.random), patch(
                "random.randint", new=rng.randint
            ), patch("random.choice", new=rng.choice), patch(
                "random.shuffle", new=rng.shuffle
            ), patch("random.sample", new=rng.sample):
                
                # Run the main loop with max_version=1 for speed
                # Train batch size 8, and train iterations 10
                train_loop.main_loop(
                    prefix=prefix,
                    families=["testing"],
                    max_workers=1,
                    max_version=1,
                    num_rounds=1,
                    batch_size=8,
                    train_iterations=10
                )

            # Check if the expected files were created
            expected_files = [
                "model-crosses-testing.0.json",
                "model-zeroes-testing.0.json",
                "model-crosses-testing.1.json",
                "model-zeroes-testing.1.json",
            ]

            created_files = os.listdir(test_dir)
            for expected_file in expected_files:
                self.assertIn(
                    expected_file, 
                    created_files, 
                    f"Expected model file {expected_file} was not created."
                )

        finally:
            # Clean up the temporary directory
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    unittest.main()
