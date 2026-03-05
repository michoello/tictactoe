import argparse
from typing import Any
from lib import train_loop
from utils import setup_logging

setup_logging("output.log")


parser = argparse.ArgumentParser(description="Train your model")
parser.add_argument("--init_model", type=str, help="Path to the initial model file")
parser.add_argument("--save_to_model", type=str, help="Path to save the trained model")
parser.add_argument("--families", type=str, nargs="+", default=["a"], help="Families to train")
parser.add_argument("--max_workers", type=int, default=1, help="Max parallel workers")
parser.add_argument("--max_version", type=int, default=4001, help="Max versions")
parser.add_argument("--num_rounds", type=int, default=4, help="Number of training rounds per version")
parser.add_argument("--train_iterations", type=int, default=100, help="Training iterations per round")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

args = parser.parse_args()

print(f"Init model: {args.init_model}")
print("Save to model:", args.save_to_model)

def main() -> None:
    train_loop.main_loop(
        prefix=args.save_to_model,
        families=args.families,
        max_workers=args.max_workers,
        max_version=args.max_version,
        num_rounds=args.num_rounds,
        train_iterations=args.train_iterations,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
