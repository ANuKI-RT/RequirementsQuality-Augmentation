import os

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
EVALUATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output", "evaluated")

def get_output_filenames():
    if os.path.exists(RESULT_DIR):
        return [f for f in os.listdir(RESULT_DIR) if os.path.isfile(os.path.join(RESULT_DIR, f))]
    else:
        raise ValueError("|X| result directory is missing")

def get_evaluated_filenames():
    if os.path.exists(EVALUATED_DIR):
        return [f for f in os.listdir(EVALUATED_DIR) if os.path.isfile(os.path.join(EVALUATED_DIR, f))]
    else:
        raise ValueError("|X| input directory is missing")