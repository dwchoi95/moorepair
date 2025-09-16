# MooRepair: Multi-Objective Optimization-based Program Repair for Programming Assignments

## Dataset

data.zip : The dataset from Codeforces.

```
|-data
    |-{problem}
    |    |-dataset.json
    |-dataset.db
    |-summary.md
```

> summary.md : Information of our datasets.  
> {problem} : Dataset folder of problem.  
> dataset.json : Raw data.  

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
4. Add LLM API Key

   ```bash
   export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
   ```

## How to Run

1. Run MooRepair as same setup
```bash
python run.py -d data/dataset.db -r -e 3
```
2. Quick Test on one problem with $pop\_size=2$, $N=2$ and multiprocessing 
```bash
python run.py -d data/dataset.db -r --problem 4 -p 2 -g 2 -m
```

## Run options
-d : The path of dataset  
--problems : flag specifies the problem (folder) name within dataset directory   
-s : Select method for selection, e.g., 'none', 'random', 'nsga3', 'rnsga3', default is 'rnsga3'  
-p : Number of population size, default is 6  
-g : Number of generations, default is 9  
-e : Number of executions, default is 1  
-r : Reset experimental results, default is false  
-m : Run multiple executions with multiprocessing, default is false  
-o : Select objective functions to considered, e.g., 'f1 f2 f3 f4 f5 f6', default is 'f1 f2 f3 f4 f5 f6'  