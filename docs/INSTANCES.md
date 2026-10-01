# Experimental Instance Set

## General

This file contain a description of the instances set made available as an algorithmic stress test for future development
on algorithms for the CPPIED or any other scientific research.

Link to the paper will be available here upon publication.

## Citation

If you use the result data available in this repository for your scientific research, please cite:

D. Richard, M. Morin, C.G. Quimper, *Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection*, European Journal of Operational Research, 2026 (being reviewed).

Latex format citation will be placed here upon publication.

## Instances Files

The instances data section contains all raw data instances and the following descriptive files:
- The configuration phase instances are listed in config_instances.txt
- The ablation analysis and empirical section instances are listed in run_instances.txt
- All instances are contained in different subdirectories of ejor_tests with the instance name as subdirectory name.

Each unique instance subdirectory contains three files in matrix format:
- The probability of detection in cppied_pod.txt
- The seabed type in cppied_problem.txt
- The required probability of detection in cppied_req.txt

We encourage for future contributors on instances the CPPIED to keep the same instance format for uniformity.

## Matrix Format

Each instance files are in matrix format and contains the following.

| Data     | Description                                                                                     |
|----------|-------------------------------------------------------------------------------------------------|
| `Header` | The first line of the file is the matrix dimension in format `m n` for m lines and n columns.   |
| `Matrix` | The second line of the file is the first row of the matrix with the space to separate entries.  |
| `Footer` | The file can terminate with `EOF`, `eof`, a line jump `\n`, or with the last row of the matrix. |

## Data types

The matrix data have the following data types:
- Floating points with three decimals cppied_pod.txt
- Integer values in {`1`,`2`,`3`} cppied_problem.txt
- Floating points with one decimal in cppied_req.txt



