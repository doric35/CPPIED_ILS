# Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection

## General

This repository contains the implementation presented in the paper:
D. Richard, M. Morin, C.G. Quimper, *Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection*, European Journal of Operational Research, 2026 (being reviewed).

Link to the paper will be available here upon publication.

## Citation

If you use this code or the data availabel in this repository for your scientific research, please cite:

Latex format citation will be placed here upon publication.

## Overview

This repository contains:
- CPPIED instances data
- The ILS heuristic C++ implementation.
- The raw csv format results.
- The data analysis scripts to reproduce figures and tables from the paper.

## Repository Structure

- `src/`              Core implementation
- `include/`          Headers
- `experiments/data/ejor`        Benchmark instances used in the paper
- `experiments/scripts`      Scripts to reproduce results
- `experiments/results`  Raw experimental results
- `external/`      Disposition of external dependencies
- `tests` Google tests
- `main.cpp` Main script to launch any of our algorithm

## Requirements

- C++ [23](https://isocpp.org)
- CMake $\geq$ [3.27](https://cmake.org)
- Gurobi [13](https://www.gurobi.com)
- LKH [2](http://webhotel4.ruc.dk/~keld/research/LKH/)
- Eigen [3.4](https://libeigen.gitlab.io)

## Compilation

```bash
git clone https://github.com/doric35/CppiedEjor.git
cd CppiedEjor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Running the Code

The command line application requires a configuration file passed as command line argument and instance parameter files
referenced from the configuration file. The above requirements need to be satisfied. The results are output in a csv file
specified in the configuration.

### Usage

```bash
./build/CppiedEJOR <configuration_file>.txt
```

### Configuration File
The configuration file in organized in a KEY=VALUE pair format. Each line specifies a pair, and characters % and # 
specifies  comments.

| Key                 | Description                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `NAME`              | Run identifier specified as <instance_id>_<algorithm_config>.                                       |
| `TIME`              | Available runtime in seconds (integer valued).                                                      |
| `SOLVER`            | Algorithm specification `ILS` or `GUROBI`                                                           |
| `ALGORITHM_CONFIG`  | Algorithm configuration `cxxxxxxxxxxx` or `cx`, where x is a configuration bit.                     |
| `WORKING_DIRECTORY` | Absolute path for generating the intermediate files.                                                |
| `SEABED_FILE`       | Absolute path to the file containing seabed information in matrix format.                           |
| `POD_FILE`          | Absolute path to the file containing probabilities of detection in matrix format.                   |
| `REQ_COVERAGE_FILE` | Absolute path to the file containing required probabilities of detection in matrix format.          |
| `VIZ_FILE`          | Absolute path to the target file for solution visualisation with .avi extension.                    |
| `CSV_LOG_FILE`      | Absolute path to the target file for logging results. Results are appended if the file exists.      |
| `SOLUTION_FILE`     | Absolute path to the target file for logging raw solution.                                          |
| `LKH_EXECUTABLE`    | Absolute path to the LKH executable file.                                                           |
| `VERBOSE`           | List of comma separated verbose specification with {`SUMMARY`,`VISUALIZE`,`ITERATIONS`,`SOLUTION`}. |

- `SUMMARY`: Log raw results.
- `VISUALIZE`: Log solution visualisation.
- `ITERATIONS`: Log iteration-wise resolution information.
- `SOLUTION`: Log raw solution.

### Algorithm configuration

The `ALGORITHM_CONFIG` field of the configuration file specifies the algorithm components to use as specified in the paper.\
When using `GUROBI`, use a single bit configuration specifying whether to use a warm-start. 

### Matrix format
Each instances parameters are specified in matrix format.\
The first line of the file specifies matrix dimensions in the format ROWS COLUMNS.\
As an example, the following describes a 3 x 4 integer valued matrix in our accepted format.

3 4\
1 2 1 1\
3 3 2 2\
1 1 2 1

The probabilities files are in the same format but contains double values with any number of digits instead of integers.


## Tested on

- macOS (Apple Clang 21)
- Linux (GCC 14.3)

## Contact

Dominik Richard\
Université Laval\
dominik.richard.1@ulaval.ca
