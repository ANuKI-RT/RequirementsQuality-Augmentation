import io
import os
import re
import json

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"input")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")

#-------------------------------------------------------------
#------ Functions to define how to read data from files ------
#-------------------------------------------------------------

#------ Functions to read from Input directory ------

def read_raw_data(files: list):
#   files is a list containing string names of files, without dir-prefix  
    data = []   
    for d in files: 
        if TRAINING_DIR in d:
            try:
                data_file = io.open(d, mode="r", encoding="utf-8")
                for line in data_file:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                        data.append(txt)
                data_file.close()
            except UnicodeDecodeError:
                data_file = io.open(d, mode="r", encoding="cp1252")
                for line in data_file:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                        data.append(txt)
                data_file.close()
        else:
            try:
                data_file = io.open(d, mode="r", encoding="utf-8")
                for line in data_file:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                        data.append(txt)
            except UnicodeDecodeError:
                data_file = io.open(d, mode="r", encoding="cp1252")
                for line in data_file:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                        data.append(txt)
                data_file.close()
    return data

def read_txt_data(files: list):
    data = []
    for d in files:
        with io.open(os.path.join(TRAINING_DIR, d), mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                txt = re.sub("\n", "", line)
                data.append(txt)
    return data

#------ Functions to read from Output directory ------

def read_output_txt_data(files: list):
    data = []
    for d in files:
        with io.open(os.path.join(RESULT_DIR, d), mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                txt = re.sub("\n", "", line)
                data.append(txt)
    return data

def read_json_files(files: list):
    for file in files:
        json_text = ""
        with io.open(os.path.join(RESULT_DIR,file), mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                json_text = json_text + line.replace("\n", "")
    return json.loads(json_text)

def comparison(data:list):

    print(len(data))
    print(len(list(set(data))))
