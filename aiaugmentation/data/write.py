from time import strftime, gmtime
import io
import os

import pandas as pd
#-------------------------------------------------------------
#------Functions to define how to write results to files------
#-------------------------------------------------------------


def write_file(results:list, classifier:str):

  x = "result"+classifier + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".txt"
  data_file=os.path.join("data", "output", x)

  with io.open(data_file, mode="w", encoding="utf-8") as outfile:
    for i, sample_output in enumerate(results):
      # The paraphrasing function returns data in tuples, where the latter number indicates the deviation of the augmented sentence from the origin prompt.
      if type(sample_output) is tuple:
        outfile.write("".join("\n{}".format(sample_output[0])))
      else:
        outfile.write("".join("\n{}".format(sample_output)))


def write_json(results:list, classifier:str):
  x = "result"+classifier+strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".json"
  data_file_path=os.path.join("data", "output", x)

  df = pd.DataFrame(results)
  df.to_json(data_file_path, indent=4, force_ascii=False, orient="values")