# MooRepair: Multi-Objective Optimization-based Program Repair for Programming Assignments

## Dataset

data.zip : The dataset from Refactory and AssignmentMender.

```
|-data
    |-benchmark.txt
    |-{problem}
    |    |-dataset.json
    |    |-log.json
    |    |-results.json
    |-...
```

> benchmark.txt : Information of each problem.
> {problem} : Dataset folder of problem.
> dataset.json : Dataset of problem which include submissions and set of test cases, etc.
> log.json : log of MooRepair.
> results.json : Each result of experiment.

## Setup

1. Environment
   `python >= 3.12`
2. Install library

   ```bash
   pip install -r requirements.txt
   ```
3. Unzip Dataset

   ```bash
   unzip data.zip
   ```

## How to Run

1. Run all problems
```bash
python run.py -d data
```

## Run options
- d : The path of dataset
- t : The timeout for compile program, default is 1sec
- a : The approach to run, e.g., 'zero', 'random', 'optimal', default is 'optimal'
- g : Number of generations, default is 10
- e : Number to execute approach, default is 1
- r : Reset results of executions, default is false
- m : Run multiple executions with multiprocessing, default is false
- o : Select objective functions to considered, e.g., 'f1 f2 f3 f4 f5 f6', default is 'f1 f2 f3 f4 f5 f6'