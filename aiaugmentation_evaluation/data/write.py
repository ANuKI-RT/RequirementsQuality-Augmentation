import pandas as pd
from time import strftime, gmtime
import os

def write_json(results:list, classifier:str):
  x = "results"+classifier+strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".json"
  data_file_path=os.path.join("data", "output", "evaluated", x)

  df = pd.DataFrame(results)
  df.to_json(data_file_path, indent=4, force_ascii=False, orient="values")
