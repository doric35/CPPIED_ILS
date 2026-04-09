import numpy as np
import sys, subprocess
import argparse
from pathlib import Path
import pandas as pd
from scipy.stats import shapiro

parser = argparse.ArgumentParser(
    prog='KAvgPerformanceCPPIED',
    description='Solve K times a instance with the same configuration for statistical analysis.',
    epilog='Pass as argument the name of the configuration file.'
)

parser.add_argument("-e", "--executable",
                    type=str,
                    required=True,
                    help="Absolute path to CPPIED executable file.")

parser.add_argument("-c", "--configuration",
                    type=str,
                    required=True,
                    help="Absolute path to configuration file.")

args = parser.parse_args()

def parse_config(file_path):
    file_config = {}

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(f"Invalid line (missing '='): {line}")

            key, value = map(str.strip, line.split("=", 1))

            # Optional: type casting
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if "," in value:
                        value = [v.strip() for v in value.split(",")]

            file_config[key] = value

    return file_config

if __name__=="__main__":
    K = 5
    config = parse_config(args.configuration)

    assert "VERBOSE" in config.keys() and "SUMMARY" in config["VERBOSE"]

    results_df = pd.read_csv(config["CSV_LOG_FILE"], delimiter=',', header=0)

    for i in range(K):
        result = subprocess.run([args.executable, args.configuration], capture_output=True, text=True)
        if result.returncode:
            #Erase last i+1 lines of the result file.
            print("Error occurred during execution, reverting the summary.")
            print("Return code:", result.returncode)
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            results_df.to_csv(config["CSV_LOG_FILE"], sep=",", index = False)
            sys.exit(1)

    new_results_df = pd.read_csv(config["CSV_LOG_FILE"], delimiter=',', header=0)
    new_results_df = new_results_df.tail(K)

    required_cols = {"length", "turns"}
    if not required_cols.issubset(new_results_df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(new_results_df.columns)}")

    if K > 3:
        print(f"Average length: {new_results_df.loc[:, 'length'].mean()}; Average turns: {new_results_df.loc[:, 'turns'].mean()}")
        print(f"Deviation length: {new_results_df.loc[:, 'length'].std()}; Deviation turns: {new_results_df.loc[:, 'turns'].std()}")
        l_stats, l_p = shapiro(new_results_df.loc[:, 'length'])
        t_stats, t_p = shapiro(new_results_df.loc[:, 'turns'])
        print(f"Normality length: {l_p}; Normality turns: {t_p}")




