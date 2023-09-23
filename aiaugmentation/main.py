import os
import gpt.gen as gen
import gpt.gpt as gpt
import data.data as data
import data.write as write
import data.read as read
import eda.eda as eda
import rtt.rttru as rttru
import para.paraphrase as pa
import experiment as ex
import evaluate

EXAMPLE_PROMPT=["MicroPython", "The system shall", "The system shall process the function as follows"]

DATA = ["0000 - cctns.txt", "0000 - gamma j.txt", "1998 - themas.txt", "2007-eirene_fun_7-2.txt", "2007-ertms.txt", "2008 - keepass.txt", "NEW - 2008 - peering.txt"]  # name of textfiles, which are used for training # "E1356-GTD-SRS-01_I1_R4.txt", "E1356-GTD-TR-01_I2_R1.txt", "RTEMS_ICD.txt", "RTEMS_SRS.txt"
RESULTS = ["resultgpt12023-07-17_20-23-04.txt", "resultEDA2023-07-27_18-43-09.txt", "resultgpt12023-07-17_16-06-20.txt", "resultgpt12023-07-17_16-11-34.txt"]

# write.write_file(data.delete_duplicates_txt(rttru.execute_rtt(read.read_raw_data(DATA), False)), "RTTen2ru")
for d in DATA:
    write.write_json(data.delete_duplicates_json(rttru.execute_rtt(read.read_raw_data([d]), True)), "RTTen2ru")

# write.write_file(data.delete_duplicates_txt(gen.execute_gpt(EXAMPLE_PROMPT, 20, False)), "GPT")
# write.write_json(data.delete_duplicates_json(gen.execute_gpt(EXAMPLE_PROMPT, 20, True)), "GPT")

# write.write_file(data.delete_duplicates_txt(eda.execute_eda(read.read_raw_data(DATA), False)), "EDA")
# write.write_json(data.delete_duplicates_json(eda.execute_eda(read.read_raw_data(DATA), True)), "EDA")

# write.write_file(data.delete_duplicates_txt(pa.execute_para(read.read_raw_data(DATA), False)), "PARA")
for d in DATA:
    write.write_json(data.delete_duplicates_json(pa.execute_para(read.read_raw_data([d]), True)), "PARA") 

# write.write_file(data.delete_duplicates_txt(rttru.execute_rtt(read.read_raw_data(DATA), False)), "RTTen2ru")
# write.write_json(data.delete_duplicates_json(rttru.execute_rtt(read.read_raw_data(DATA), True)), "RTTen2ru")

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