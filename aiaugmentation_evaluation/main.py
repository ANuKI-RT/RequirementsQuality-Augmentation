import evaluate
import data.write as write
import data.data as data

def evaluate_experiment(input:str):
    write.write_json(evaluate.experiment_evaluation(input),"EVALUATION")

def evaluation_results_calculation(input:str):
    evaluate.evaluation_results_calculation(input)

def get_output_files():
    return data.get_output_filenames()

def get_evaluated_files():
    return data.get_evaluated_filenames()

