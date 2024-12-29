from fairseq.models.transformer import TransformerModel
import os
import re
import spacy
import json
# import fairseq as fairseq
from rtt.fairseq import fairseq as fairseq
# import fairseq.fairseq.logging.meters

EN_DE_PATH = os.path.join("rtt","wmt19.en-de.joined-dict.ensemble")
DE_EN_PATH = os.path.join("rtt","wmt19.de-en.joined-dict.ensemble")

def load_spacy_model():
    return spacy.load("en_core_web_lg")

def check_special_words(doc):
    modified_tokens = []
    entities_and_oov = []
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
        else:
            modified_tokens.append(token.text)

    return modified_tokens, entities_and_oov

def process_line(line, nlp):
    doc = nlp(line['line'])
    modified_tokens, entities_and_oov = check_special_words(doc)
    
    modified_sentence = " ".join(modified_tokens)

    return {
        "file": line['file'],
        "original": line['line'],
        "modified": modified_sentence,
        "entities_and_oov": entities_and_oov,
    }

def write_output_file(output_file_path, processed_data):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        json.dump(processed_data, output_file, indent=4)

    print(f"Processed file saved to: {output_file_path}")

def rtt_augmentation(input_file_path):
    nlp = load_spacy_model()
    
    base_name = os.path.basename(input_file_path)
    name, ext = os.path.splitext(base_name)
    
    print(input_file_path)

    if os.path.isfile(input_file_path):
        """Read a JSON file and return its content as a list of dictionaries."""
        results = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate the structure
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            for line in data:
                processed_data = process_line(line, nlp)
                result = execute_rtt_experimentv2(processed_data)
                results.append(result)
        else:
            raise ValueError("The JSON file does not contain a valid list of dictionaries.")    
    else:
        print("No")

    return results

def execute_rtt_experimentv2(d, gen_json:bool = True):
    results = []
    MODALITY_VERBS = {"shall", "should", "must", "may"}
    '''
    Underlying transformer model is also from: Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.
        https://fairseq.readthedocs.io/en/v0.9.0/_modules/fairseq/models/transformer.html
    '''
    en2de = TransformerModel.from_pretrained(
        model_name_or_path=EN_DE_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=EN_DE_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(EN_DE_PATH, "bpecodes")
    )

    de2en = TransformerModel.from_pretrained(
        model_name_or_path=DE_EN_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=DE_EN_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(DE_EN_PATH, "bpecodes")
    )
    
    for verb in MODALITY_VERBS:
        d["modified"] = re.sub(f"\\b{verb}\\b", f"<preserve>{verb}</preserve>", d["modified"], flags=re.IGNORECASE)
    
    entities_and_oov_scope= len(d["entities_and_oov"])
    
    tokens = en2de.encode(d["modified"])
    output = en2de.generate(tokens, beam=3, nbest=3, skip_invalid_size_inputs=True)
    ger_samples = [en2de.decode(x["tokens"])for x in output]
    res_rtt =[]
    for r in ger_samples:
        res = []
        tokens = de2en.encode(r)
        output = de2en.generate(tokens, beam=3, nbest=3, skip_invalid_size_inputs=True)
        back_translated_output = [de2en.decode(x["tokens"])for x in output]
        
        for b in back_translated_output:
            b = re.sub(r"<preserve>(.*?)</preserve>", r"\\1", b)
            i = 0
            while i < entities_and_oov_scope:
                pattern = rf"<\s*{re.escape(str(i))}\s*>"
                print(pattern)
                if re.search(pattern, b):
                    replacement = ""
                    for r in d["entities_and_oov"]:
                        if r["position"] == i:
                            replacement = r["token"]
                    re.sub(pattern, replacement, b)
                i+=1
            res.append(b)
        
        d["model"] = "RTT"
        d["augmented"] = res
        if gen_json == True:
            res_rtt.append(d)
        '''for b in back_translated_output:
            if gen_json == True:
                res_rtt.append(["RTT",d, b])
            else:
                if type(b) is list:
                    for bb in b:
                        res_rtt.append(bb)
                else:
                    res_rtt.append(b)    '''
        results.append(res_rtt)

    if gen_json == False:
        results = res_rtt  

    return results
