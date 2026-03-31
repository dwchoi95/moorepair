# MooRepair: Multi-Objective Optimization-based Program Repair for Programming Assignments

<!-- ![image](./overview.png) -->
<div align="center">
<img src="https://anonymous.4open.science/r/moorepair-BA14/overview.png" 
     alt="overview"
     style="width:clamp(320px, 50%, 900px); height:auto; display:block;" />
</div>

## Setup

1. Environment
   `Ubuntu`
   `python >= 3.13`

2. (Optional) Virtual Environment
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install Packages

   ```bash
   pip install -r requirements.txt
   ```

4. Load Dataset

   ```bash
   python dataset.py build --language "Python 3"
   python dataset.py verify
   python dataset.py summary
   ```

5. LLM API Key Setting

   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY="your_api_key_here"
   ```


## How to Run
0. Fast Run

   ```bash
   python run.py -d data/670_B -s
   ```

1. MooRepair (GPT-3.5-Turbo)

   ```bash
   python run.py -d data -a MooRepair -g 4 -p 6 -l gpt-3.5-turbo
   ```

2. MooRepair (GPT-5-nano)

   ```bash
   python run.py -d data -a MooRepair -g 4 -p 6 -l gpt-5-nano
   ```

3. PaR+EffiLearner (GPT-3.5-Turbo)

   ```bash
   python run.py -d data -a PaREL -g 5 -p 6 -l gpt-3.5-turbo
   ```

4. PaR+EffiLearner (GPT-5-nano)

   ```bash
   python run.py -d data -a PaREL -g 5 -p 6 -l gpt-5-nano
   ```

5. MooRepair: Random Selection (GPT-3.5-Turbo)

   ```bash
   python run.py -d data -a Random -g 4 -p 6 -l gpt-3.5-turbo
   ```

6. MooRepair: Random Selection (GPT-5-nano)

   ```bash
   python run.py -d data -a Random -g 4 -p 6 -l gpt-5-nano
   ```

## Run Options

| Option | Long Option     | Description                                     | Default        |
|--------|-----------------|-------------------------------------------------|----------------|
| `-d`   | `--dataset`     | Path to dataset directory or JSON file          | (required)     |
| `-a`   | `--approach`    | Approach to run: `PaREL`, `Random`, `MooRepair` | `MooRepair`    |
| `-g`   | `--generations` | Number of generations                           | `4`            |
| `-p`   | `--popsize`     | Population size                                 | `6`            |
| `-l`   | `--llm`         | LLM model name                                  | `gpt-3.5-turbo`|
| `-t`   | `--temperature` | LLM sampling temperature                        | `0.8`          |
| `-s`   | `--sampling`    | Use 10% sampling of buggy programs              | `False`        |
| `-r`   | `--reset`       | Reset experiments results                       | `False`        |
