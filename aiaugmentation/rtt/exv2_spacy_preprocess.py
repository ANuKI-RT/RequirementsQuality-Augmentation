import os
import spacy
import json
import re

def load_spacy_model():
    return spacy.load("en_core_web_lg")

def read_txt_file(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    return lines

def read_json_file(input_file_path):
    """Read a JSON file and return its content as a list of dictionaries."""
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate the structure
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    else:
        raise ValueError("The JSON file does not contain a valid list of dictionaries.")

def check_special_words(doc, modality_verbs):
    modified_tokens = []
    entities_and_oov = []
    modality_details = []
    i = 0
    for token in doc:
        if token.is_oov or token.ent_type_:
            modified_tokens.append(f"<{i}>")
            entities_and_oov.append({
                "token": token.text,
                "position": i,
                "start": token.idx,
                "end": token.idx + len(token.text)
            })
            i+=1
        elif token.text.lower() in modality_verbs:
            modality_details.append({
                "token": token.text,
                "position": i,
                "start": token.idx,
                "end": token.idx + len(token.text)
            })
            modified_tokens.append(token.text)
        else:
            modified_tokens.append(token.text)

    return modified_tokens, entities_and_oov, modality_details

def process_line(line, nlp, modality_verbs):
    doc = nlp(line['line'])
    modified_tokens, entities_and_oov, modality_details = check_special_words(doc, modality_verbs)
    
    modified_sentence = " ".join(modified_tokens)

    return {
        "file": line['file'],
        "original": line['line'],
        "modified": modified_sentence,
        "entities_and_oov": entities_and_oov,
        "modality_verbs": modality_details
    }

def write_output_file(output_file_path, processed_data):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(processed_data, output_file, indent=4)

    print(f"Processed file saved to: {output_file_path}")

def preprocess_json_file(input_file_path):
    modality_verbs = {"shall", "should", "must", "may"}
    
    nlp = load_spacy_model()
    
    base_name = os.path.basename(input_file_path)
    name, ext = os.path.splitext(base_name)
    
    print(input_file_path)

    if os.path.isfile(input_file_path):
        lines = read_json_file(input_file_path)
        results = []
        
        for line in lines:
            result = process_line(line, nlp, modality_verbs)
            results.append(result)
    else:
        print("No")

    return results