"""
    Prints stats of the datasets in the /CoreferenceEntitiesInReqs folder.
    
    The stats shall be helpful for improving the communication of the experiment.

    Taken from https://gitlab.reutlingen-university.de/anuki/reqmeasurementworkspace/-/blob/main/CoreferenceEntitiesInReqs/data/data_statistics.py?ref_type=heads, with slight modifications to parse json
"""

import os
import statistics as st
import json

# Print stats for dataset 1 - ESA
num_aug_PURE_projects = 0
num_aug_PURE_reqs = 0
lengths_aug_PURE_reqs = []
lengths_aug_PURE_words = []

print("~+~+~+~+~+~+~+~+~+~    Dataset 1    ~+~+~+~+~+~+~+~+~+~")
print(" **** Augmented Requirements from PURE Dataset ****")

for doc in os.listdir("./output"):
    if doc not in "experiments":
        print("")
        print("       - " + doc.replace(".json", ""))

        num_aug_PURE_projects += 1


        with open("./output" + "/" + doc, "r", encoding="utf-8") as f:
            file_contents = json.loads(f.read())

            num_reqs_per_doc = 0
            lenghts_reqs_per_doc = []
            lenghts_words_per_req = []
            
            lines = f.readlines()
            for req in file_contents:
                line = req[2]
                num_reqs_per_doc += 1
                    
                words_req = line.split(" ")
                for word in words_req:
                    lenghts_words_per_req.append(len(word))
                    lengths_aug_PURE_words.append(len(word))
                    
                lenghts_reqs_per_doc.append(len(line.split(" ")))
                lengths_aug_PURE_reqs.append(len(line.split(" ")))
                
            print("          " + str(num_reqs_per_doc) + " requirements extracted.")
            print("          " + str(round(st.mean(lenghts_reqs_per_doc), 2)) + " words per requirement in average.")
            print("          " + str(st.median(lenghts_reqs_per_doc)) + " median words per requirement.")
            print("          " + str(max(lenghts_reqs_per_doc)) + " maximal words per requirement.")
            print("          " + str(round(st.mean(lenghts_words_per_req), 2)) + " chars per word in average.")
            print("          " + str(int(st.median(lenghts_words_per_req))) + " median chars per word.")
            print("          " + str(max(lenghts_words_per_req)) + " maximal chars per word.")
            
            num_aug_PURE_reqs += num_reqs_per_doc
            f.close()

# Print overall stats
print("")
print("~+~+~+~+~+~+~+    Overall Statistics    ~+~+~+~+~+~+~+")
print("")
print("       " + str(num_aug_PURE_reqs) + " extracted requirements from " + str(num_aug_PURE_projects/2) + " PURE projects.")
print("       -> " + str(round(st.mean(lengths_aug_PURE_reqs), 2)) + " words per requirement in average.")
print("       -> " + str(int(st.median(lengths_aug_PURE_reqs))) + " median words per requirement.")
print("       -> " + str(max(lengths_aug_PURE_reqs)) + " maximal words per requirement.")
print("       -> " + str(round(st.mean(lengths_aug_PURE_words), 2)) + " chars per word in average.")
print("       -> " + str(int(st.median(lengths_aug_PURE_words))) + " median chars per word.")
print("       -> " + str(max(lengths_aug_PURE_words)) + " maximal chars per word.")
