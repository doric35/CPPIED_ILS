# Experimental Results

## General

This file contain a description of the raw results that we made available to 
replicate the experimental section of the following paper:

D. Richard, M. Morin, C.G. Quimper, *Iterated Local Search for Coverage Path Planning with Imperfect Extended Detection*, European Journal of Operational Research, 2026 (being reviewed).

Link to the paper will be available here upon publication.

## Citation

If you use the result data available in this repository for your scientific research, please cite:

Latex format citation will be placed here upon publication.

## Results Files

The experimental results section contains all files to replicate the tables and figure from the Manuscript:
- The aggregated ablation analysis results in ejor_ablation_path.csv
- The raw ablation analysis results in ejor_ablation_results.csv
- The raw configuration phase results in their order of execution in ejor_configuration_results.csv
- The raw empirical analysis result in ejor_results.csv

## Raw Results Structure

Each results file is in csv format with the following columns.

| Key      | Description                                                            |
|----------|------------------------------------------------------------------------|
| `name`   | Run identifier specified as <instance_id>_<algorithm_config>.          |
| `solver` | Resolution method in {`ILS`,`GUROBI`}.                                 |
| `length` | Path length of the incumbent.                                          |
| `turns`  | Number of turns of the incumbent.                                      |
| `time`   | Resolution time in seconds.                                            |                                   
| `status` | Resolution status in {`OPTIMAL`,`SUBOPTIMAL`,`TIME_LIMIT_INFEASIBLE`}. |
