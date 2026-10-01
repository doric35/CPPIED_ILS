# Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection

## General

This repository contains the implementation presented in the paper:
D. Richard, M. Morin, C.G. Quimper, *Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection*, European Journal of Operational Research, 2026 (being reviewed).

Link to the paper will be available here upon publication.

## Citation

If you use this code or the data available in this repository for your scientific research, please cite:

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

## Running the code

### Compilation

```bash
git clone https://github.com/doric35/CppiedEjor.git
cd CppiedEjor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The command line application requires a configuration file passed as command line argument and instance parameter files
referenced from the configuration file. The above requirements need to be satisfied. The results are output in a csv file
specified in the configuration.

### Usage

```bash
./build/CppiedEJOR <configuration_file>.txt
```

### Documentation

- [Detailed results](docs/RESULTS.md)
- [Instances data](docs/INSTANCES.md)
- [Algorithm configuration](docs/ALGORITHMS.md)

## Tested on

- macOS (Apple Clang 21)
- Linux (GCC 14.3)

## Contact

Dominik Richard\
Université Laval\
dominik.richard.1@ulaval.ca
