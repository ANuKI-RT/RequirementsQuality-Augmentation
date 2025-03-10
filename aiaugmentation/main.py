import os
import gpt.gen as gen
import gpt.gpt as gpt
import data.data as data
import data.write as write
import data.read as read
import data.exv2_data as readv2
import eda.eda as eda
# import rtt.rttru as rttru
import rtt.rttde as rttde
import rtt.rttdeExpv2 as rttdeV2
import rtt.exv2_spacy_preprocess as pre
import rtt.exv2_spacy_postprocess as post
import para.paraphrase as pa
# import experiment as ex
import evaluate

EXAMPLE_PROMPT=["MicroPython", "The system shall", "The system shall process the function as follows"]
DATA = ["demo.txt"]
RESULTS = ["resultPARA2023-09-23_15-03-33.json", "resultPARA2023-09-23_15-06-49.json", "resultPARA2023-09-23_15-12-15.json", "resultPARA2023-09-23_15-59-14.json", "resultPARA2023-09-23_16-14-09.json", "resultPARA2023-09-23_16-16-02.json", "resultPARA2023-09-23_16-17-05.json"]
TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","input","ESA_data")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data","output")

# write.write_file(data.delete_duplicates_txt(rttru.execute_rtt(read.read_raw_data(DATA), False)), "RTTen2ru")
# write.write_json(data.delete_duplicates_json(rttde.execute_rtt(read.read_raw_data(DATA), True)), "RTTen2de")
# write.write_json(data.delete_duplicates_json(pa.execute_para(read.read_raw_data(DATA), True)), "PARA") 
# write.write_json(data.delete_duplicates_json(eda.execute_eda(read.read_raw_data(DATA), True)), "EDA")
# write.write_file(data.delete_duplicates_json(gen.execute_gpt(read.read_raw_data(DATA), True)), "GPT")
# write.write_json(rttde.execute_rtt_experimentv2(pre.preprocess_json_file(os.path.join(RESULT_DIR,"experimentV2_input","experimentV2.json"))), "RTTen2de")
# write.write_json(eda.execute_eda_experimentv2(read.read_json_data(os.path.join(RESULT_DIR,"experimentV2_input","experimentV2.json"))),"EDA")
# write.write_json(rttdeV2.rtt_augmentation(os.path.join(RESULT_DIR,"experimentV2_input","experimentV2.json")), "RTTen2de")

# write.write_file(data.delete_duplicates_txt(gen.execute_gpt(EXAMPLE_PROMPT, 20, False)), "GPT")
# write.write_json(data.delete_duplicates_json(gen.execute_gpt(EXAMPLE_PROMPT, 20, True)), "GPT")

# write.write_file(data.delete_duplicates_txt(eda.execute_eda(read.read_raw_data(DATA), False)), "EDA")
# write.write_json(data.delete_duplicates_json(eda.execute_eda(read.read_raw_data(DATA), True)), "EDA")

# write.write_file(data.delete_duplicates_txt(pa.execute_para(read.read_raw_data(DATA), False)), "PARA")
# write.write_json(data.delete_duplicates_json(pa.execute_para(read.read_raw_data(DATA), True)), "PARA")

# write.write_file(data.delete_duplicates_txt(rttde.execute_rtt(read.read_raw_data(DATA), False)), "RTTen2de")
# write.write_json(data.delete_duplicates_json(rttde.execute_rtt(read.read_raw_data(DATA), True)), "RTTen2de")

# gpt.gpt(read.read_raw_data(["Test.txt"]))
# write.write_file(data.delete_duplicates_txt(gen.execute_gpt(EXAMPLE_PROMPT, 20, False)), "GPT")
# write.write_json(data.delete_duplicates_json(gen.execute_gpt(EXAMPLE_PROMPT, 20, True)), "GPT")

# write.write_file(data.delete_duplicates_txt(eda.execute_eda(read.read_raw_data(["Test.txt"],), False)), "EDA")
# write.write_json(data.delete_duplicates_json(eda.execute_eda(read.read_raw_data(["Test.txt"],), True)), "EDA")

# write.write_file(data.delete_duplicates_txt(pa.execute_para(read.read_raw_data(["Test.txt"],), False)), "PARA")
# write.write_json(data.delete_duplicates_json(pa.execute_para(read.read_raw_data(["Test.txt"],), True)), "PARA") 

#-----------------------------------------------------------------------------------------------------
#------------------------------------------Train AI Models--------------------------------------------
#-----------------------------------------------------------------------------------------------------

# gpt.gpt(read.read_raw_data(DATA))

#-----------------------------------------------------------------------------------------------------
#------------------------------------------Conduct Experiment-----------------------------------------
#-----------------------------------------------------------------------------------------------------
'''
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["PARA","EDA"],1, "Experiment1ParaEda")
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["PARA","GPT"],1, "Experiment1ParaGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["RTT","EDA"],1, "Experiment1RttEda")
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["RTT","GPT"],1, "Experiment1RttGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["PARA","EDA"],2, "Experiment2ParaEda")
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["PARA","GPT"],2, "Experiment2ParaGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["RTT","EDA"],2, "Experiment2RttEda")
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["RTT","GPT"],2, "Experiment2RttGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["PARA","EDA"],3, "Experiment3ParaEda")
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["PARA","GPT"],3, "Experiment3ParaGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["RTT","EDA"],3, "Experiment3RttEda")
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["RTT","GPT"],3, "Experiment3RttGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["PARA","EDA"],4, "Experiment4ParaEda")
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["PARA","GPT"],4, "Experiment4ParaGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["RTT","EDA"],4, "Experiment4RttEda")
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["RTT","GPT"],4, "Experiment4RttGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["PARA","EDA"],5, "Experiment5ParaEda")
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["PARA","GPT"],5, "Experiment5ParaGpt")
'''

'''
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["RTT","EDA"],5, "Experiment5RttEda")
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["RTT","GPT"],5, "Experiment5RttGpt")
'''
'''
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["PARA-RTT","EDA-GPT"],1, "Experiment1Para-Rtt_Eda-Gpt")
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["RTT-PARA","GPT-EDA"],1, "Experiment1Rtt-Para_Gpt-Eda")

ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["PARA-RTT","EDA-GPT"],2, "Experiment2Para-Rtt_Eda-Gpt")
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["RTT-PARA","GPT-EDA"],2, "Experiment2Rtt-Para_Gpt-Eda")

ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["PARA-RTT","EDA-GPT"],3, "Experiment3Para-Rtt_Eda-Gpt")
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["RTT-PARA","GPT-EDA"],3, "Experiment3Rtt-Para_Gpt-Eda")

ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["PARA-RTT","EDA-GPT"],4, "Experiment4Para-Rtt_Eda-Gpt")
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["RTT-PARA","GPT-EDA"],4, "Experiment4Rtt-Para_Gpt-Eda")

ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["PARA-RTT","EDA-GPT"],5, "Experiment5Para-Rtt_Eda-Gpt")
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["RTT-PARA","GPT-EDA"],5, "Experiment5Rtt-Para_Gpt-Eda")

ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["PARA-RTT","GPT-EDA"],1, "Experiment1Para-Rtt_Gpt-Eda")
ex.augment_data_predefined(read.read_raw_data(["experiment1.txt"]), 100, 70, ["RTT-PARA","EDA-GPT"],1, "Experiment1Rtt-Para_Eda-Gpt")

ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["PARA-RTT","GPT-EDA"],2, "Experiment2Para-Rtt_Gpt-Eda")
ex.augment_data_predefined(read.read_raw_data(["experiment2.txt"]), 100, 70, ["RTT-PARA","EDA-GPT"],2, "Experiment2Rtt-Para_Eda-Gpt")

ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["PARA-RTT","GPT-EDA"],3, "Experiment3Para-Rtt_Gpt-Eda")
ex.augment_data_predefined(read.read_raw_data(["experiment3.txt"]), 100, 70, ["RTT-PARA","EDA-GPT"],3, "Experiment3Rtt-Para_Eda-Gpt")

ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["PARA-RTT","GPT-EDA"],4, "Experiment4Para-Rtt_Gpt-Eda")
ex.augment_data_predefined(read.read_raw_data(["experiment4.txt"]), 100, 70, ["RTT-PARA","EDA-GPT"],4, "Experiment4Rtt-Para_Eda-Gpt")

ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["PARA-RTT","GPT-EDA"],5, "Experiment5Para-Rtt_Gpt-Eda")
ex.augment_data_predefined(read.read_raw_data(["experiment5.txt"]), 100, 70, ["RTT-PARA","EDA-GPT"],5, "Experiment5Rtt-Para_Eda-Gpt")
'''