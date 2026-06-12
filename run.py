import os
import glob
import argparse

from src.approaches import Experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="Path to dataset directory or JSON file")
    parser.add_argument('-a', '--approach', type=str, default="MooRepair",
                        choices=["PaREL", "MooRepair"],
                        help="Approach to run (default: MooRepair). "
                             "PaREL also records PaR-only results in the same run")
    parser.add_argument('-ab', '--ablation', type=str, default=None,
                        choices=["random", "no_crossover", "no_mutation",
                                 "rand_survivor", "rand_strategy",
                                 "rand_pairing", "no_early_stop"],
                        help="Ablate a single MooRepair component; 'random' "
                             "replaces ALL selection steps with random choices "
                             "(only valid with -a MooRepair)")
    parser.add_argument('-g', '--generations', type=int, default=4,
                        help="Number of generations (default: 4)")
    parser.add_argument('-p', '--popsize', type=int, default=6,
                        help="Population size (default: 6)")
    parser.add_argument('-l', '--llm', type=str, default='gpt-3.5-turbo',
                        help="LLM model name in LiteLLM format, e.g. gpt-3.5-turbo, "
                             "ollama/codellama, ollama/gemma2, hosted_vllm/<repo>, "
                             "gemini/gemma-3-27b-it (default: gpt-3.5-turbo)")
    parser.add_argument('-ap', '--api-base', type=str, default=None,
                        help="API base URL for self-hosted servers, e.g. "
                             "http://localhost:11434 (Ollama) or "
                             "http://localhost:8000/v1 (vLLM)")
    parser.add_argument('-t', '--temperature', type=float, default=0.8,
                        help="LLM temperature (default: 0.8)")
    parser.add_argument('-to', '--timeout', type=int, default=60,
                        help="Per-call LLM timeout in seconds; raise for "
                             "slow local models, e.g. 600 (default: 60)")
    parser.add_argument('-s', '--sampling', action='store_true', default=False,
                        help="Use 10%% sampling of buggy programs")
    parser.add_argument('-r', '--reset', action='store_true', default=False,
                        help="Reset overall.csv before running experiments")
    args = parser.parse_args()

    assert os.path.isfile(args.dataset) or os.path.isdir(args.dataset), \
        "Dataset path does not exist"
    assert args.generations > 0, "Generations must be a positive integer"
    assert args.popsize > 0, "Population size must be a positive integer"
    assert args.ablation is None or args.approach == "MooRepair", \
        "--ablation is only valid with -a MooRepair"

    problems = []
    if os.path.isdir(args.dataset):
        problems = glob.glob(os.path.join(args.dataset, '**', '*.json'), recursive=True)
    else:
        problems.append(args.dataset)

    ex = Experiments(
        approach=args.approach,
        generations=args.generations,
        pop_size=args.popsize,
        llm=args.llm,
        temperature=args.temperature,
        timeout=args.timeout,
        sampling=args.sampling,
        reset=args.reset,
        ablation=args.ablation,
        api_base=args.api_base
    )
    ex.run(problems)
