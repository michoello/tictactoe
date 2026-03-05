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

args = parser.parse_args()

print(f"Init model: {args.init_model}")
print("Save to model:", args.save_to_model)

def main() -> None:
    train_loop.main_loop(
        prefix=args.save_to_model,
        families=args.families,
        max_workers=args.max_workers,
        max_version=args.max_version
    )

if __name__ == "__main__":
    main()
