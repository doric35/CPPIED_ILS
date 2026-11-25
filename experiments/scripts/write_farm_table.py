from argparse import ArgumentParser
from pathlib import Path
import os

parser = ArgumentParser()
parser.add_argument("-s", "--source", type=Path, required=True, help="Source directory containing all instances run configurations.")
parser.add_argument("-t", "--target", type=Path, required=True, help="Target filename for meta farm table file.")
parser.add_argument("-e", "--executable", type=Path, required=True, help="Absolute path to the executable.")
parser.add_argument("-n", "--nruns", type=int, default=5, help="Number of run for each instance.")
args = parser.parse_args()

def get_instance_runs_lines(instanceconfig : Path):
    lines = [f"{args.executable} {instanceconfig}" for _ in range(args.nruns)]
    return lines

def get_instance_runs():
    lines = []
    for directory, subdirectories, files in os.walk(args.source):
        if "configuration.txt" in files:
            lines.extend(get_instance_runs_lines(Path(os.path.join(directory, "configuration.txt"))))
    return lines

def write_runs(runs: list):
   args.target.write_text("\n".join(runs))

if __name__ == "__main__":
    runs = get_instance_runs()
    write_runs(runs)

