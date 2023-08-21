import experiment as ex
import evaluate
import data.write as write
import data.read as read
import data.data as data
import rtt.rttru as rttru
import eda.eda as eda
import para.paraphrase as pa
import gpt.gen as gpt
import gpt.gpt as train_gpt

def get_input_files():
    return data.get_input_filenames()

def get_output_files():
    return data.get_output_filenames

def create_experiment(files:list, total_amount:int, pos_ratio:float, models:list):
    write.write_json(ex.augment_data(data.get_random_raw_data(files=files, total_amount=total_amount), pos_ratio=pos_ratio, models=models), "EXPERIMENT")

def evaluate_experiment(file:str):
    evaluate.experiment_evaluation(file)

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- RTT ----------------------------------------------
#--------------------------------------------------------------------------------------------------

def rtt_txt(input: list, gen_json: bool):
    write.write_file(data.delete_duplicates_txt(rttru.execute_rtt(read.read_raw_data(input), gen_json)), "RTT")

def rtt_json(input: list, gen_json: bool):
    write.write_json(data.delete_duplicates_json(rttru.execute_rtt(read.read_raw_data(input), gen_json)), "RTT")

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- EDA ----------------------------------------------
#--------------------------------------------------------------------------------------------------

def eda_txt(input: list, gen_json: bool):
    write.write_file(data.delete_duplicates_txt(eda.execute_eda(read.read_raw_data(input), gen_json)), "EDA")

def eda_json(input: list, gen_json: bool):
    write.write_json(data.delete_duplicates_json(eda.execute_eda(read.read_raw_data(input), gen_json)), "EDA")

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- PARA ---------------------------------------------
#--------------------------------------------------------------------------------------------------

def para_txt(input: list, gen_json: bool):
    write.write_file(data.delete_duplicates_txt(pa.execute_para(read.read_raw_data(input), gen_json)), "PARA")

def para_json(input: list, gen_json: bool):
    write.write_json(data.delete_duplicates_json(pa.execute_para(read.read_raw_data(input), gen_json)), "PARA") 

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- GPT ----------------------------------------------
#--------------------------------------------------------------------------------------------------

def gpt_txt(input: list, gen_json: bool, amount:int=20):
    write.write_file(gpt.execute_gpt(input, amount, gen_json), "GPT")

def gpt_json(input: list, gen_json: bool, amount:int=20):
    write.write_json(gpt.execute_gpt(input, amount, gen_json), "GPT")

def train_gpt(input: list):
    train_gpt.gpt(read.read_raw_data(input))