from argparse import ArgumentParser
from pathlib import Path
import os
import pandas as pd

parser = ArgumentParser()
parser.add_argument("-s", "--source", type=Path, required=True, help="Path to the experiments_directory.")
parser.add_argument("-t", "--target", type=Path, required=True, help="Path to target result file.")
parser.add_argument("-n", "--nruns", type=int, default=5, help="Keep the nruns last values.")
args = parser.parse_args()

HEADER = ["name","solver","length","turns","time","status"]

def read_target_results():
    results = []
    for directory, subdirectories, files in os.walk(args.target):
        if "log_results.csv" in files:
            filename = Path(os.path.join(directory, "log_results.csv"))
            results.append(pd.read_csv(filename))
            results[-1] = results[-1].tail(5)
    return pd.concat(results)

if __name__ == "__main__":
    res = read_target_results()
    res.columns = HEADER
    if args.target.exists():
        base = pd.read_csv(args.target)
        res = pd.concat([res, base], ignore_index=True)

    res.drop_duplicates(subset="name", keep="last" ,inplace=True, ignore_index=True)
    res.to_csv(args.target, index=False)