from .parrotTest import Parrot
import re

import warnings
warnings.filterwarnings("ignore")

def execute_para(data:list, gen_json:bool = False):
  results = []
  counter = 0
  parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

  for d in data:
    paraphrases = parrot.augment(input_phrase=d, use_gpu=False, do_diverse=True, max_return_phrases=10, adequacy_threshold=0.9, fluency_threshold=0.9)

    if gen_json == True:
      if paraphrases == None: 
        counter += 1
      else:
        par_results = []
        for p in paraphrases:
          if p is None:
            continue
          else:
            par_results.append(["PARA", re.sub("/n", "",d), p[0]])
        results.append(par_results)
    else:  
      if paraphrases == None: 
        counter += 1 
      else:
        for p in paraphrases:
          results.append(p)
  print("|I| Total skipped values (no paraphrase found): ", counter)
  return results

# Method with adjusted output format for the experiment, the combined-parameter enables the functionality to process data from another model.
# only one result per prompt is permitted, it is chosen by going for the maximum difference value between original prompt and augmented sentence, indicated by the second value in the returned tuple after succesful paraphrasing.

def execute_para_experiment(data:list, combined:bool = False):
  results = []
  counter = 0
  parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

  if combined == False:
    for d in data:
      paraphrases = parrot.augment(input_phrase=d, use_gpu=False, do_diverse=True, max_return_phrases=10, adequacy_threshold=0.9, fluency_threshold=0.9)

      if paraphrases == None: 
        counter += 1
      else:
        max_p = max(paraphrases)[1]
        max_ps = []
        for p in paraphrases:
          if p is None:
            continue
          else:
            # checking for maximum difference value in returned tuple
            if p[1] == max_p:
              max_ps.append(p)
        if len(max_ps)>0:
          results.append(["PARA", re.sub("/n", "",d), max_ps[0][0]])
  else:
    for d in data:
      paraphrases = parrot.augment(input_phrase=d[2], use_gpu=False, do_diverse=True, max_return_phrases=10, adequacy_threshold=0.9, fluency_threshold=0.9)
      if paraphrases == None: 
        counter += 1
      else:
        max_p = max(paraphrases)[1]
        max_ps = []
        for p in paraphrases:
          if p is None:
            continue
          else:
            # checking for maximum difference value in returned tuple
            if p[1] == max_p:
              max_ps.append(p)
        if len(max_ps)>0:
          results.append(["PARA", re.sub("/n", "",d[1]), max_ps[0][0]])

  print("|I| Total skipped values (no paraphrase found): ", counter)
  return results

# The parrot package is implemented from https://github.com/PrithivirajDamodaran/Parrot_Paraphraser