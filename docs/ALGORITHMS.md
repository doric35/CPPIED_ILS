# Algorithm Configuration

## Configuration File
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

## Algorithm configuration

The `ALGORITHM_CONFIG` field of the configuration file specifies the algorithm components to use as specified in the paper.\
When using `GUROBI`, use a single bit configuration specifying whether to use a warm-start. 