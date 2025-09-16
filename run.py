import os
import argparse

from experiments import Experiments
from src.genetic.selector import SELECTIONS
from src.genetic.fitness import Fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="The path of dataset")
    parser.add_argument('--problems', nargs='+', default=None,
                        help="flag specifies the problem (folder) name within dataset directory")
    parser.add_argument('-a', '--amount', type=int, default=100,
                        help="Experiment with n% of data, default is 100%")
    parser.add_argument('-g', '--generations', type=int, default=9,
                        help="Number of generations, default is 9")
    parser.add_argument('-p', '--popsize', type=int, default=6,
                        help="Number of population size, default is 6")
    parser.add_argument('-s', '--selection', type=str, default='rnsga3',
                        help=f"Select method for selection, e.g., {', '.join(SELECTIONS)}")
    parser.add_argument('--threshold', type=float, default=1.0,
                        help="Set a threshold(0~1) for early stop of GP, default is 1.0")
    parser.add_argument('-e', '--executions', type=int, default=1,
                        help="Number of executions, default is 1")
    parser.add_argument('-r', '--reset', action='store_true', default=False,
                        help="Reset experimental results")
    parser.add_argument('-m', '--multiprocess', action='store_true', default=False,
                        help="Run with multiprocessing")
    parser.add_argument('-o', '--objectives', nargs='+', default=" ".join(Fitness.OBJECTIVES),
                        help=f"Select objectives to considered, e.g., '{" ".join(Fitness.OBJECTIVES)}'")
    
    args = parser.parse_args()


    dataset = args.dataset
    problems = args.problems
    amount = args.amount
    generations = args.generations
    pop_size = args.popsize
    selection = args.selection.lower()
    threshold = args.threshold
    executions = args.executions
    reset = args.reset
    multi = args.multiprocess
    objectives = args.objectives
    if isinstance(objectives, str):
        objectives = objectives.split(' ')
    
    
    assert os.path.isfile(dataset), "Database file doesn't exist"
    assert 1 <= amount <= 100, \
        "Dataset amount must be between 10 and 100"
    assert isinstance(generations, int) and generations > 0, \
        "Generations must be a positive integer"
    assert isinstance(pop_size, int) and pop_size > 0, \
        "Population size must be a positive integer"
    if isinstance(selection, str) and selection not in SELECTIONS:
        raise ValueError(f"Invalid selection method, choose {', '.join(SELECTIONS)}")
    assert 0 <= threshold <= 1, \
        "Threshold must be between 0 and 1"
    assert isinstance(executions, int) and executions > 0, \
        "Executions must be a positive integer"
    assert isinstance(reset, bool), \
        "Reset must be a boolean value"
    assert isinstance(multi, bool), \
        "Multiprocess must be a boolean value"
    assert isinstance(objectives, list) and \
        all(isinstance(obj, str) for obj in objectives), \
        "Objectives must be a list of strings"
    
    
    ex = Experiments(
        dataset, amount,
        generations, pop_size, 
        selection, threshold, 
        objectives, executions, 
        reset, multi)
    ex.run(problems)
    # ex.update_exp(problems)
