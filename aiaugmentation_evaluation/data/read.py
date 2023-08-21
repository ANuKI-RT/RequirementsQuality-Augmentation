import io
import re
import json
import os

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
EVALUATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output","evaluated")

def read_json_files(files: list):
    for file in files:
        json_text = ""
        with io.open(os.path.join(RESULT_DIR,file), mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                line = re.sub("\n", "", line)
                json_text = json_text + line
    return json.loads(json_text)

def read_evaluated_json_files(files: list):
    for file in files:
        json_text = ""
        with io.open(os.path.join(EVALUATED_DIR,file), mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                line = re.sub("\n", "", line)
                json_text = json_text + line
    return json.loads(json_text)