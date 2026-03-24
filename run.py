import os
import glob
import argparse

from src.baselines import Experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Path to dataset directory or JSON file")
    parser.add_argument('-a', '--approach', type=str, default="moorepair",
                        choices=["moorepair", "baseline"],
                        help="Approach to run (default: moorepair)")
    parser.add_argument('-g', '--generations', type=int, default=5,
                        help="Number of generations (default: 5)")
    parser.add_argument('-p', '--popsize', type=int, default=5,
                        help="Population size (default: 5)")
    parser.add_argument('-s', '--selection', action='store_true', default=False,
                        help="Use random selection")
    parser.add_argument('-l', '--llm', type=str,
                        default='codellama/CodeLlama-7b-Instruct-hf',
                        help="LLM model name (default: CodeLlama-7b-Instruct-hf)")
    parser.add_argument('--temperature', type=float, default=0.8,
                        help="LLM sampling temperature (default: 0.8)")
    parser.add_argument('--sampling', action='store_true', default=False,
                        help="Use 10%% sampling of buggy programs")
    parser.add_argument('-r', '--reset', action='store_true', default=False,
                        help="Reset existing results")
    args = parser.parse_args()

    assert os.path.isfile(args.dataset) or os.path.isdir(args.dataset), \
        "Dataset path does not exist"
    assert args.generations > 0, "Generations must be a positive integer"
    assert args.popsize > 0, "Population size must be a positive integer"

    problems = []
    if os.path.isdir(args.dataset):
        problems = glob.glob(os.path.join(args.dataset, '**', '*.json'), recursive=True)
    else:
        problems.append(args.dataset)

    ex = Experiments(
        generations=args.generations,
        pop_size=args.popsize,
        selection=args.selection,
        llm=args.llm,
        temperature=args.temperature,
        sampling=args.sampling,
        approach=args.approach.lower(),
        reset=args.reset,
    )
    ex.run(problems)
