import argparse
from typing import Any
from lib import train_loop

# Duplicate all output to a file
import sys
class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
    def write(self, data: Any) -> None:
        for s in self.streams:
            s.write(data)
    def flush(self) -> None:
        for s in self.streams:
            s.flush()

logfile = open("output.log", "w")  # TODO: cli arg?
sys.stdout = Tee(sys.stdout, logfile)

parser = argparse.ArgumentParser(description="Train your model")
parser.add_argument("--init_model", type=str, help="Path to the initial model file")
parser.add_argument("--save_to_model", type=str, help="Path to save the trained model")

args = parser.parse_args()

print(f"Init model: {args.init_model}")
print("Save to model:", args.save_to_model)

def main() -> None:
    train_loop.main_loop(args=args)

if __name__ == "__main__":
    main()
