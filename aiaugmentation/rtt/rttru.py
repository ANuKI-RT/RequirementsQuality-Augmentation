from fairseq.models.transformer import TransformerModel
import os
# import fairseq as fairseq
from rtt.fairseq import fairseq as fairseq
# import fairseq.fairseq.logging.meters

# contains the model configuration for russian as intermediate language

def execute_rtt(data:list, gen_json:bool = False):
    EN_RU_PATH = os.path.join("rtt","wmt19.en-ru.ensemble")
    RU_EN_PATH = os.path.join("rtt","wmt19.ru-en.ensemble")
    results = []
    res = []
    '''
    Underlying transformer model is also from: Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.
        https://fairseq.readthedocs.io/en/v0.9.0/_modules/fairseq/models/transformer.html
    '''
    en2ru = TransformerModel.from_pretrained(
        model_name_or_path=EN_RU_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=EN_RU_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(EN_RU_PATH, "bpecodes")
    )

    ru2en = TransformerModel.from_pretrained(
        model_name_or_path=RU_EN_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=RU_EN_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(RU_EN_PATH, "bpecodes")
    )
    
    for d in data: 
        tokens = en2ru.encode(d)
        output = en2ru.generate(tokens, beam=5, nbest=5, skip_invalid_size_inputs=True)
        rus_samples = [en2ru.decode(x["tokens"])for x in output]
        res_rtt =[]

        for r in rus_samples:

            tokens = ru2en.encode(r)
            output = ru2en.generate(tokens, beam=5, nbest=5, skip_invalid_size_inputs=True)
            back_translated_output = [ru2en.decode(x["tokens"])for x in output]

            for b in back_translated_output:
                if gen_json == True:
                    res_rtt.append(["RTT",d, b])
                else:
                    if type(b) is list:
                        for bb in b:
                            res_rtt.append(bb)
                    else:
                        res_rtt.append(b)    

        results.append(res_rtt)

    if gen_json == False:
        results = res_rtt  

    return results


# Method with adjusted output format for the experiment, the combined-parameter indicates if the output will be further processed in another model afterwards.

def execute_rtt_experiment(data:list, combined: bool = False):
    EN_RU_PATH = os.path.join("rtt","wmt19.en-ru.ensemble")
    RU_EN_PATH = os.path.join("rtt","wmt19.ru-en.ensemble")
    results = []
    res = []
    '''
    Underlying transformer model is also from: Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
        <https://arxiv.org/abs/1706.03762>`_.
        https://fairseq.readthedocs.io/en/v0.9.0/_modules/fairseq/models/transformer.html
    '''
    en2ru = TransformerModel.from_pretrained(
        model_name_or_path=EN_RU_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=EN_RU_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(EN_RU_PATH, "bpecodes")
    )

    ru2en = TransformerModel.from_pretrained(
        model_name_or_path=RU_EN_PATH,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        data_name_or_path=RU_EN_PATH,
        bpe='fastbpe',
        tokenizer='moses',
        bpe_codes = os.path.join(RU_EN_PATH, "bpecodes")
    )

    if combined == False:
        for d in data: 
            tokens = en2ru.encode(d)
            output = en2ru.generate(tokens, beam=5, nbest=1, skip_invalid_size_inputs=True)
            rus_samples = [en2ru.decode(x["tokens"])for x in output]
            back_translated_output = []

            for r in rus_samples:

                tokens = ru2en.encode(r)
                output = ru2en.generate(tokens, beam=5, nbest=1, skip_invalid_size_inputs=True)
                back_translated_output += [ru2en.decode(x["tokens"])for x in output]


            results.append(["RTT",d, back_translated_output[0]])
    else:
        for d in data: 
            tokens = en2ru.encode(d[2])
            output = en2ru.generate(tokens, beam=5, nbest=1, skip_invalid_size_inputs=True)
            rus_samples = [en2ru.decode(x["tokens"])for x in output]
            back_translated_output = []

            for r in rus_samples:

                tokens = ru2en.encode(r)
                output = ru2en.generate(tokens, beam=5, nbest=1, skip_invalid_size_inputs=True)
                back_translated_output += [ru2en.decode(x["tokens"])for x in output]


            results.append(["RTT",d[1], back_translated_output[0]])

    return results