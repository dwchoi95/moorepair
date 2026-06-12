"""Measure the syntax pass/fail rate of initial candidate generation.

Runs ONLY MooRepair's initialization stage (Variation.correct) for each
buggy program — no evolutionary search — costing exactly pop_size LLM
calls per buggy (initialization is single-attempt; call-level failures
are retried up to 3 times inside Models.run). Counts are appended to
init_stats.csv; analyze afterwards with:

  python stats.py --init

Usage:
  python init_check.py -d data/670_B -s          # one problem, 10% sampling
  python init_check.py -d data -l gpt-3.5-turbo  # full dataset
"""

import os
import glob
import argparse

from tqdm import tqdm

from src.approaches.moorepair import MooRepair, INIT_STATS_PATH
from src.llms import Models, Tokenizer
from src.utils import Loader
from src.execution import Tester

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Path to dataset directory or JSON file")
    parser.add_argument('-p', '--popsize', type=int, default=6,
                        help="Population size (default: 6)")
    parser.add_argument('-l', '--llm', type=str, default='gpt-3.5-turbo',
                        help="LLM model name in LiteLLM format, e.g. gpt-3.5-turbo, "
                             "ollama/codellama (default: gpt-3.5-turbo)")
    parser.add_argument('-ap', '--api-base', type=str, default=None,
                        help="API base URL for self-hosted servers")
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                        help="LLM temperature (default: 0.8)")
    parser.add_argument('-to', '--timeout', type=int, default=60,
                        help="Per-call LLM timeout in seconds (default: 60)")
    parser.add_argument('-s', '--sampling', action='store_true', default=False,
                        help="Use 10%% sampling of buggy programs")
    parser.add_argument('-r', '--reset', action='store_true', default=False,
                        help="Reset init_stats.csv before measuring")
    args = parser.parse_args()

    assert os.path.isfile(args.dataset) or os.path.isdir(args.dataset), \
        "Dataset path does not exist"

    problems = []
    if os.path.isdir(args.dataset):
        problems = glob.glob(os.path.join(args.dataset, '**', '*.json'), recursive=True)
    else:
        problems.append(args.dataset)

    Models.set(model=args.llm, temperature=args.temperature,
               timeout=args.timeout, api_base=args.api_base)
    if not args.llm.startswith("gpt-"):
        Tokenizer.set(args.llm)
    if args.reset and os.path.exists(INIT_STATS_PATH):
        os.remove(INIT_STATS_PATH)

    loader = Loader(args.sampling)
    for problem in problems:
        assignment, timelimit, memlimit, buggys, references, testcases = \
            loader.run(problem)
        problemId = assignment['id'].replace("/", "_")
        print(f"\n=== {problemId} ({len(buggys)} buggys) ===")

        Tester.init_globals(testcases, timelimit, memlimit)
        moo = MooRepair(buggys, references, assignment)
        for buggy in tqdm(buggys, desc="Buggy", position=0):
            moo._init_population(buggy, args.popsize)

    print(f"\nSaved: {INIT_STATS_PATH}")
    print("Analyze with: python stats.py --init")
