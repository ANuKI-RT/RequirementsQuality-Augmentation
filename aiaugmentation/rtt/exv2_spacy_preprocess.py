import os
import spacy
import json

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

def check_oov_and_entities(doc):
    modified_tokens = []
    entities_and_oov = []
    i = 0
    for token in doc:
        if token.is_oov or token.ent_type_:
            modified_tokens.append(f"[[[[[{i}]]]]]")
            entities_and_oov.append({
                "token": token.text,
                "position": i,
                "start": token.idx,
                "end": token.idx + len(token.text)
            })
            i+=1
        else:
            modified_tokens.append(token.text)

    return modified_tokens, entities_and_oov

def check_modality_verbs(doc, modality_verbs):
    modality_details = []

    for token in doc:
        if token.text.lower() in modality_verbs:
            modality_details.append({
                "token": token.text,
                "start": token.idx,
                "end": token.idx + len(token.text)
            })

    return modality_details

def process_line(line, nlp, modality_verbs):
    doc = nlp(line['line'])
    modified_tokens, entities_and_oov = check_oov_and_entities(doc)
    modality_details = check_modality_verbs(doc, modality_verbs)
    
    modified_sentence = " ".join(modified_tokens)

    return {
        "file": line['file'],
        "original": line['line'],
        "modified": modified_sentence,
        "entities_and_oov": entities_and_oov,
        "modality_verbs": modality_details
    }
    
'''def process_text_file(lines, nlp, modality_verbs):
    processed_data = []

    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespaces
        if not line:
            continue  # Skip empty lines
        processed_data.append(process_line(line, nlp, modality_verbs))

    return processed_data'''

def write_output_file(output_file_path, processed_data):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(processed_data, output_file, indent=4)

    print(f"Processed file saved to: {output_file_path}")

def preprocess_json_file(input_file_path):
    
    output_dir = "spacy_preprocessed"
    input_dir = "processed_files"
    modality_verbs = {"shall", "should", "must", "may"}
    output_suffix = "_preprocessed.json"
    
    nlp = load_spacy_model()
    
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(os.path.join(input_dir, input_file_path))
    name, ext = os.path.splitext(base_name)
    output_file_path = os.path.join(output_dir, f"{name}{output_suffix}")
    
    input_file_path = os.path.join(input_dir, input_file_path)
    
    if os.path.isfile(input_file_path):
        lines = read_json_file(input_file_path)
        results = []
        
        for line in lines:
            result = process_line(line, nlp, modality_verbs)
            results.append(result)
    else:
        print("No")

    write_output_file(output_file_path, results)