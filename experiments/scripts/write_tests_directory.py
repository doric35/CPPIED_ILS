from argparse import ArgumentParser
from pathlib import Path
from itertools import product

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=Path, required=True)
parser.add_argument("-i", "--instances", type=Path, required=True)
parser.add_argument("-d", "--directory", type=Path, required=True)
parser.add_argument("-s", "--source", type=Path, required=True)
parser.add_argument("--lkh", type=Path, required=True)
args = parser.parse_args()

SIZE_TIME_MAP = {"s1616" : 60, "s6464" : 600, "s128128" : 1200}

def extract_size(c : str):
    return c.split("_")[0]

def write_configuration(c : dict, filename : Path):
    lines = []
    for k,v in c.items():
        lines.append(f"{k}={v}")

    filename.write_text("\n".join(lines))

def create_environment(c : str, i : str):
    run_name = f"{i}_{c}"
    run_dir = Path(args.directory / run_name)
    run_dir.mkdir()

    config_name = run_dir / "configuration.txt"

    configuration = dict()
    configuration["NAME"] = str(run_dir.stem)
    configuration["TIME"] = SIZE_TIME_MAP[extract_size(i)]
    configuration["SOLVER"] = "ILS"
    configuration["ALGORITHM_CONFIG"] = c
    configuration["WORKING_DIRECTORY"] = str(run_dir)
    configuration["SEABED_FILE"] = str(args.source / i / "cppied_problem.txt")
    configuration["POD_FILE"] = str(args.source / i / "cppied_pod.txt")
    configuration["REQ_COVERAGE_FILE"] = str(args.source / i / "cppied_req.txt")
    configuration["VERBOSE"] = "SUMMARY"
    configuration["LKH_EXECUTABLE"] = str(args.lkh)
    configuration["CSV_LOG_FILE"] = str(run_dir / "results_log.csv")

    write_configuration(configuration, config_name)

def read_config(filename : Path):
    configs = filename.read_text().split()
    return configs

if __name__=="__main__":
    configs = read_config(args.config)
    instances = read_config(args.instances)
    for c, i in product(configs, instances):
        create_environment(c, i)