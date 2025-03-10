import argparse
from main import *

def main():
    args = argparse.ArgumentParser(description='Use Ai Augmentation Techniques')
    args.add_argument("-m", "--method", required=True, type=str, help="Select method to execute -- the following methods are available: \n show_output_files, \n show_evaluated_files, \n evaluate_experiment, \n calc_evaluate_experiment")
    args.add_argument("-i","--input", required=False, type=str, help="input file -- evaluate_experiment and calc_evaluate_experiment both need the name of the file without path but with file-extension")

    args = args.parse_args()

    if args.method == "show_output_files":
        print(get_output_files())

    if args.method == "show_evaluated_files":
        print(get_evaluated_files())

    if args.method == "evaluate_experiment" and args.input != None:
        evaluate_experiment(args.input)
    elif args.method == "evaluate_experiment" and args.input == None:
        print("Please add a file on which you want to conduct the evaluation.")

    if args.method == "calc_evaluate_experiment" and args.input != None:
        evaluation_results_calculation(args.input)
    elif args.method == "calc_evaluate_experiment" and args.input == None:
        print("Please add a evaluated experiment file on which you want to conduct the calculation.")
    

if __name__ == "__main__":
    # calling the main function
    main()