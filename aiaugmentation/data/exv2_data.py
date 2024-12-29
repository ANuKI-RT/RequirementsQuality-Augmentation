import os
import random
from collections import defaultdict
from difflib import SequenceMatcher
import json
import re

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"input","ESA_data")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output", "experimentV2_input")

def extract_txt_files(directory):
    """Extract all .txt files from the directory."""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".txt")
    ]

def group_files_by_similarity(files, threshold=0.8):
    """Group files based on similarity in filenames."""
    groups = []

    for file in files:
        added = False
        for group in groups:
            if SequenceMatcher(None, os.path.basename(file), os.path.basename(group[0])).ratio() > threshold:
                group.append(file)
                added = True
                break
        if not added:
            groups.append([file])

    return groups

def read_all_lines_from_group(group):
    """Read all lines from all files in the group."""
    all_lines = []
    for file in group:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                    all_lines.append({"line": txt.strip(), "file": os.path.basename(file)})
                f.close()
        except UnicodeDecodeError:
            with open(file, "r", encoding="cp1252") as f:
                for line in lines:
                    if line.strip():
                        if re.search("\[SEP\]", line) is None:
                            txt = re.sub("\[END\]", "", line.replace("\n", ""))
                        else:
                            txt = re.sub("\[SEP\].*?\[END\]", "", line.replace("\n", ""))
                    all_lines.append({"line": txt.strip(), "file": os.path.basename(file)})
                f.close()
    return all_lines

def process_directory(file, directory=TRAINING_DIR, num_lines=20):
    """Extract and process all .txt files in a directory."""
    files = extract_txt_files(directory)
    file_groups = group_files_by_similarity(files)
    
    results = []
    
    rnum = int(num_lines/len(file_groups))
    print(rnum)
    
    for group in file_groups:
        all_lines = read_all_lines_from_group(group)
        random.shuffle(all_lines)
        results.extend(all_lines[:int(rnum)])
        
    with open(os.path.join(RESULT_DIR,file), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {os.path.join(RESULT_DIR,file)}")
    