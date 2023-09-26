import os
import io
import re
import sys
import json

import seaborn as sns
import numpy as np
import pandas as pd
import random
import data.read as read
import data.write as write

import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
'''NB. If you publish work that uses NLTK, please cite the NLTK book as follows:
    Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc.'''

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"input")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")

def get_output_filenames():
    if os.path.exists(RESULT_DIR):
        return [os.path.join(RESULT_DIR,f) for f in os.listdir(RESULT_DIR) if os.path.isfile(os.path.join(RESULT_DIR, f))]
    else:
        raise ValueError("|X| result directory is missing")

def get_input_filenames():
    if os.path.exists(TRAINING_DIR):
        return [os.path.join(TRAINING_DIR,f) for f in os.listdir(TRAINING_DIR) if os.path.isfile(os.path.join(TRAINING_DIR, f))]
    else:
        raise ValueError("|X| input directory is missing")

def get_random_raw_data(files: list, total_amount: int = 100):
    results = []
    training_data = read.read_raw_data(files)

    random.shuffle(training_data)

    if len(training_data) > total_amount:
        random_sample = random.sample(training_data, total_amount)
    else:
        random_sample = random.sample(training_data, len(training_data))

    for r in random_sample:
        r = re.sub("\n", "", r)
        results.append(r)

    return results

def get_test_data(amount: int, files: list = None):
    results = []
    data = []
    if list == None:
        files = [file for file in os.listdir(RESULT_DIR) if file.endswith(".json")]

    data.append(read.read_json_files(files))

    i = 0
    while i < amount:
        file_rand = np.random.randint(0, len(data))
        file_selection = data[file_rand]

        prompt_rand = np.random.randint(0, len(file_selection))
        data_selection = file_selection[prompt_rand]

        selection_rand = np.random.randint(0, len(data_selection))
        choice = data_selection[selection_rand]

        results.append(choice)
        i += 1

    return results

def count_training_data(classifier: list):
    files = []
    counter = 0
    for file in os.listdir(RESULT_DIR):
        if file.endswith(".json"):
            for c in classifier:
                if re.search(c, file):
                    files.append(file)

    data = read.read_json_files(files)
    for d in data:
        counter += len(d)
    print(len(data))
    return counter

def delete_duplicates_txt(data: list):
    clear_data = []
    for x in data: 
        if x not in clear_data:
            clear_data.append(x)
    print("|I| Total data: ", len(data))
    print("|I| Total processed data: ", len(clear_data))
    print("|I| Total removed duplicates: ", len(data)-len(clear_data))
    return clear_data

def delete_duplicates_json(data: list):
    total_data = 0
    processed_data = 0
    results = []

    for req in data:
        clear_data = []
        prompt = ""
        classifier = ""
        for i in req:
            classifier = i[0]
            prompt = i[1]
            if i[2].lower() != prompt.lower():
                clear_data.append(i[2])
        clear_data = set(clear_data)
        for c in clear_data:
            results.append([classifier, prompt, c])

        total_data += len(req)
        processed_data += len(clear_data)

    print("|I| Total data: ", total_data)
    print("|I| Total final data: ", processed_data)
    print("|I| Total removed duplicates: ", total_data - processed_data)
    return results

def delete_duplicates_json2(data: list):
    results = []
    temp = []
    for req in data:
        augmented_reqs = []
        classifier = "",
        prompt = ""
        standardized_augmented_req = " ".join(req[2].split()).lower().rstrip(".").rstrip("-").rstrip("''")
        if standardized_augmented_req != " ".join(req[1].split()).lower().rstrip(".").rstrip("-").rstrip("''"):
            if standardized_augmented_req not in temp:
                temp.append(standardized_augmented_req)
                augmented_reqs.append(req[2])
                classifier = req[0]
                prompt = req[1]
        for _ in augmented_reqs:
            results.append([classifier, prompt, _])
    return results

def test(data: list):
    i = 0
    temp2 = []
    for req in data:
        print("-"*200)
        print("-"*200)
        print("-"*200)
        temp = []
        standardized_augmented_req = " ".join(req[2].split()).lower().rstrip(".").rstrip("-").rstrip("''")

        print("/")
        print(standardized_augmented_req)
        print(" ".join(req[1].split()).lower())
        print("/")

        if standardized_augmented_req != " ".join(req[1].split()).lower().rstrip(".").rstrip("-").rstrip("''"):
            if standardized_augmented_req not in temp2:
                temp.append(standardized_augmented_req)
                temp2.append(standardized_augmented_req)
                i += 1
        
        print(temp)
        print(i)
    print(len(temp2))

# This method compares the augmentation results with their corresponding prompts. If it is strictily the same, it will be removed.
def delete_duplicates_experiment(data: list, classifier:str, number:int):
    results = []
    duplicates = []

    for d in data:
        print(d)
        prompt = d[1]
        sentence = d[2]

        if prompt != sentence:
            results.append(d)
        else:
            duplicates.append(d)
    
    document = []
    document.append("".join(("results for duplicate removal for method: ", str(classifier), " in experiment no. ", str(number))))
    document.append("".join(("|I| Total data: ", str(len(data)))))
    document.append("".join(("|I| Total final data: ", str(len(results)))))
    document.append("".join(("|I| Total removed duplicates: ", str(len(data) - len(results)))))
    document.append("".join(("-"*100)))
    document.append("".join(("List of duplicates: ")))
    document.append(duplicates)

    write.write_file(document, "".join(("DUPLICATEREMOVAL_", str(classifier))))
    return results


'''def delete_duplicates(files: list):
    data_list = read_json_files(files)
    data_set = set(data_list)

    print(len(data_list))
    print(len(data_set))
    print("Total duplicates removed: ",len(data_list)-len(data_set))

    return list(data_set)'''

        

# This function is used, to get a random selection of each of the given Trainingdata documents:
'''def get_random_selection(files: list, div: int):
    ex = []
    data = []
    for d in files:
        with io.open(d, mode="r", encoding="utf-8") as data_file:
            for line in data_file:
                ex.append(line)
    amount = len(ex) / div
    print(amount)
    i = 0
    while i < amount:
        x = np.random.randint(0, len(ex))
        print(x)
        data.append(ex[x])
        i += 1
    return data'''

# To augment for experiment, take a random subset made of samples from different given files and return it for further processing.
'''def get_random_trainingdata(files: list, total_amount: int = 100): CAN BE REMOVED IF OTHER GET_RANDOM_TRAINING_DATA IS BETTER
    results = []
    data = read.read_raw_data(files)
    i = 0
    while i < total_amount:
        sample = random.choice(data)
        if len(data) > total_amount:
            if sample not in results:
                sample = sample.strip()
                results.append(sample)
                i+=1
        else:
            results = data
            break
    return results'''

#   This function is used to check for the token distribution and their corresponding length. 
'''def get_training_data(DATA: list):
    doc_lengths=[]

    for d in DATA:
        tokens = nltk.word_tokenize(d)
        doc_lengths.append(len(tokens))

    doc_lengths = np.array(doc_lengths)
    sns.displot(doc_lengths)
    plt.show()
    print(np.average(doc_lengths))
    print(doc_lengths)'''