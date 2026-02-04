import os
import glob
import argparse

from experiments import Experiments
from src.genetic.selection import SELECTIONS
from src.genetic.fitness import Fitness

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help="The path of dataset")
    parser.add_argument('-g', '--generations', type=int, default=20,
                        help="Number of generations, default is 20")
    parser.add_argument('-p', '--popsize', type=int, default=10,
                        help="Number of population size, default is 10")
    parser.add_argument('-i', '--initialization', action='store_true', default=False,
                        help="Use initialization for population")
    parser.add_argument('-s', '--selection', type=str, default='nsga3',
                        help=f"Select method for selection, e.g., {', '.join(SELECTIONS)}")
    parser.add_argument('--threshold', type=int, default=5,
                        help="Set a threshold for early stop of GP, default is 5")
    parser.add_argument('-l', '--llm', type=str, default='codellama/CodeLlama-7b-Instruct-hf',
                        help="Select LLM model for generating patches, default is codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument('--temperature', type=float, default=0.8,
                        help="Set temperature for LLM model, default is 0.8")
    parser.add_argument('-t', '--timelimit', type=int, default=1,
                        help="Time limit for each test case in seconds, default is 1")
    parser.add_argument('-e', '--executions', type=int, default=1,
                        help="Number of executions, default is 1")
    parser.add_argument('--sampling', action='store_true', default=False,
                        help="Experiment with 10% sampling data")
    parser.add_argument('-r', '--reset', action='store_true', default=False,
                        help="Reset experimental results")
    parser.add_argument('-m', '--multiprocess', action='store_true', default=False,
                        help="Run with multiprocessing")
    parser.add_argument('-o', '--objectives', nargs='+', default=" ".join(Fitness.OBJECTIVES),
                        help=f"Select objectives to considered, e.g., '{' '.join(Fitness.OBJECTIVES)}'")
    args = parser.parse_args()


    dataset = args.dataset
    generations = args.generations
    pop_size = args.popsize
    initialization = args.initialization
    selection = args.selection.lower()
    threshold = args.threshold
    llm = args.llm
    temperature = args.temperature
    timelimit = args.timelimit
    executions = args.executions
    sampling = args.sampling
    reset = args.reset
    multi = args.multiprocess
    objectives = args.objectives
    if isinstance(objectives, str):
        objectives = objectives.split(' ')
    

    assert os.path.isfile(dataset) or os.path.isdir(dataset), "Dataset doesn't exist"
    assert isinstance(generations, int) and generations > 0, \
        "Generations must be a positive integer"
    assert isinstance(pop_size, int) and pop_size > 0, \
        "Population size must be a positive integer"
    assert isinstance(initialization, bool), \
        "Initialization must be a boolean value"
    if isinstance(selection, str) and selection not in SELECTIONS:
        raise ValueError(f"Invalid selection method, choose {', '.join(SELECTIONS)}")
    assert isinstance(threshold, int) and threshold >= 1, \
        "Threshold must be a positive integer"
    assert isinstance(executions, int) and executions > 0, \
        "Executions must be a positive integer"
    assert isinstance(sampling, bool), \
        "Sampling must be a boolean value"
    assert isinstance(reset, bool), \
        "Reset must be a boolean value"
    assert isinstance(multi, bool), \
        "Multiprocess must be a boolean value"
    assert isinstance(objectives, list) and \
        all(isinstance(obj, str) for obj in objectives), \
        "Objectives must be a list of strings"
    
    # dataset 디렉토리 부터 하위의 모든 .json 파일들 수집해서 prblems 리스트에 저장
    problems = []
    if os.path.isdir(dataset):
        json_files = glob.glob(os.path.join(dataset, '**', '*.json'), recursive=True)
        problems.extend(json_files)
    else:
        problems.append(dataset)
    
    ex = Experiments(
        generations, pop_size, initialization,
        selection, threshold, 
        llm, temperature, timelimit,
        objectives, executions, 
        sampling, reset, multi
    )
    ex.run(problems)
    
