import data.data as data
import rtt.rttru as rtt
import eda.eda as eda
import para.paraphrase as para
import numpy as np
import gpt.gen as gpt
import random
import data.write as wr
import nltk

def augment_data_random(raw_data: list, total_amount: int = 100, pos_ratio: float = 0.7, models: list = ["EDA","PARA","GPT","RTT"], gen_json: bool = True):
    results = []

    random.shuffle(raw_data)

    if pos_ratio > 0:
        total_pos_amount = raw_data[0:int(len(raw_data)*(pos_ratio))]
        pos_results = augment_pos_data(total_pos_amount, models, gen_json)
        random.shuffle(pos_results)
        results += pos_results[0:int(len(raw_data)*(pos_ratio))]

    if pos_ratio < 1: 
        total_neg_amount = raw_data[int(len(raw_data)*(pos_ratio)):int(len(raw_data))]
        neg_results = augment_neg_data(total_neg_amount, models, gen_json)
        random.shuffle(neg_results)
        results += neg_results[int(len(raw_data)*(pos_ratio)):int(len(raw_data))]

    random.shuffle(results)
    return results[0:total_amount-1]

def augment_pos_data(raw_data: list, models: list, gen_json: bool):
    results = []
    
    para_check = False
    rtt_check = False
    if "PARA" in models:
          para_check = True
    if "RTT" in models:
        rtt_check = True

    if para_check and rtt_check == True:
        data_size = np.random.randint(1, len(raw_data))

        rtt_results = rtt.execute_rtt(raw_data, gen_json)
        for r in rtt_results:
            if r:
                results += r
        para_results = para.execute_para(raw_data, gen_json)
        for r in para_results:
            if r:
                results += r
    elif para_check == True:
            results += para.execute_para(raw_data, gen_json)
    elif rtt_check == True:
            results += rtt.execute_rtt(raw_data, gen_json)
    else: 
          print("|X| No supported Model for positive data generation chosen.")
    
    return results

def augment_neg_data(raw_data: list, models: list, gen_json: bool):
    results = []

    gpt_check = False
    eda_check = False
    if "GPT" in models:
          gpt_check = True
    if "EDA" in models:
          eda_check = True

    if eda_check and gpt_check == True:
        data_size = np.random.randint(1, len(raw_data))
        gpt_results = gpt.execute_gpt(prompt=["The system shall", "The Program should", "The system shall include the following methods", "The program shall execute the"], num_return_sequences=30, gen_json=gen_json)
        for r in gpt_results:
             if r:
                  results += r
        eda_results = eda.execute_eda(data=raw_data, gen_json=gen_json)
        for r in eda_results:
             if r:
                  results += r
    elif gpt_check == True:
            results += gpt.execute_gpt(["The system shall", "The Program should", "The system shall include the following methods", "The program shall execute the"],len(raw_data), gen_json)
    elif eda_check == True:
            results += eda.execute_eda(data=raw_data, gen_json=gen_json)
    else: 
          print("|X| No supported Model for negative data generation chosen.")
    
    return results


def augment_data_predefined(raw_data: list, total_amount: int = 100, pos_ratio: int = 70, models: list = ["EDA","PARA","GPT","RTT"], experiment_no: int = 1, filename:str = "EXPERIMENT"):
    results = []

    if "PARA" in models:
        results += data.delete_duplicates_experiment(data=para.execute_para_experiment(raw_data[:pos_ratio]), classifier="PARA", number=experiment_no)

    if "RTT" in models:
        results += data.delete_duplicates_experiment(data=rtt.execute_rtt_experiment(raw_data[:pos_ratio]), classifier="RTT", number=experiment_no)

    if "EDA" in models:
        results += data.delete_duplicates_experiment(data=eda.execute_eda_experiment(raw_data[pos_ratio:]), classifier="EDA", number=experiment_no)

    if "GPT" in models:
        results += data.delete_duplicates_experiment(data=gpt.execute_gpt_experiment(raw_data[pos_ratio:], 1), classifier="GPT", number=experiment_no)

    if "PARA-RTT" in models:
        results += data.delete_duplicates_experiment(para.execute_para_experiment(rtt.execute_rtt_experiment(raw_data[:pos_ratio]), True), classifier="PARA-RTT", number=experiment_no)

    if "RTT-PARA" in models:
        results += data.delete_duplicates_experiment(rtt.execute_rtt_experiment(para.execute_para_experiment(raw_data[:pos_ratio]), True), classifier="RTT-PARA", number=experiment_no)
    
    if "GPT-EDA" in models:
        results += data.delete_duplicates_experiment(gpt.execute_gpt_experiment(eda.execute_eda_experiment(raw_data[pos_ratio:]), 1, True), classifier="GPT-EDA", number=experiment_no)

    if "EDA-GPT" in models:
        results += data.delete_duplicates_experiment(eda.execute_eda_experiment(gpt.execute_gpt_experiment(raw_data[pos_ratio:], 1), True), classifier="EDA-GPT", number=experiment_no)


    random.shuffle(results)

    wr.write_json(results[0:total_amount], filename)

    print("--"*100)
    print("COMPLETED ONE RUN, ", filename)
    print("--"*100)


def augment_data_predefined_with_duplicates(raw_data: list, total_amount: int = 100, pos_ratio: int = 70, models: list = ["EDA","PARA","GPT","RTT"], filename:str = "EXPERIMENT"):
    results = []

    if "PARA" in models:
        results += para.execute_para_experiment(raw_data[:pos_ratio])

    if "RTT" in models:
        results += rtt.execute_rtt_experiment(raw_data[:pos_ratio])

    if "EDA" in models:
        results += eda.execute_eda_experiment(raw_data[pos_ratio:])

    if "GPT" in models:
        results += gpt.execute_gpt_experiment(raw_data[pos_ratio:], 1)
    
    if "PARA-RTT" in models:
        results += para.execute_para_experiment(rtt.execute_rtt_experiment(raw_data[:pos_ratio]), True)

    if "RTT-PARA" in models:
        results += rtt.execute_rtt_experiment(para.execute_para_experiment(raw_data[:pos_ratio]), True)
    
    if "GPT-EDA" in models:
        results += gpt.execute_gpt_experiment(eda.execute_eda_experiment(raw_data[pos_ratio:]), 1, True)

    if "EDA-GPT" in models:
        results += eda.execute_eda_experiment(gpt.execute_gpt_experiment(raw_data[pos_ratio:], 1), True)

    random.shuffle(results)
    
    len(results)

    wr.write_json(results[0:total_amount], filename)

    print("--"*100)
    print("COMPLETED ONE RUN")
    print("--"*100)
    
    