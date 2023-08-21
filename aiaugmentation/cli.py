import argparse
from cli_functions import *

args = argparse.ArgumentParser(description='Use Ai Augmentation Techniques')
args.add_argument("-m", "--method", required=True, type=str, help="Select method to execute")
args.add_argument("-i","--input", nargs='+', required=False, type=str, help="input file")
args.add_argument("-json", "--gen_json", action='store_true', help="Specify if the output type shall be in .json. This format provides the samples with additional information (corresponding prompt and used model) as opposed to the .txt output, which generates a list of samples. Default is False")

args.add_argument("-a","--amount", required=False, type=int, help="Amount of desired augmented results, default is 100 for creation of experiment, 20 for generated gpt prompts.")
args.add_argument("-r", "--pos_ratio", required=False, type=float, help="Ratio for desired correct augmented results, ratio for corresponding negative results is calculated. Pick a value 0 =< n =< 1. Default is 0.7")
args.add_argument("-mo","--models", required=False, type=list, help="The models to be used for augmentation. For semantically correct augmented samples, Paraphrasing (enter: PARA) or Round-Trip Translation (enter: RTT) may be choosen. For semantically incorrect augmented samples, Easy Data Augmentation (enter EDA) or Text generation based on GPT (enter: GPT) may be choosen. Default is [\"EDA\", \"GPT\", \"RTT\", \" PARA\"]")

args = args.parse_args()

if args.method == "show_input_files":
    print(get_input_files())

if args.method == "show_output_files":
    print(get_output_files)

if  args.method == "create_experiment" and args.input != None:
    create_experiment(files=args.input, total_amount=args.amount, pos_ratio=args.pos_ratio, models=args.models)

if args.method == "evaluate_experiment" and args.input != None:
    evaluate_experiment(args.input)

if args.method == "get_gen_json":
    print(args.gen_json)

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- RTT ----------------------------------------------
#--------------------------------------------------------------------------------------------------

if args.method == "rtt" and args.input != None:
    if args.gen_json == False:
        rtt_txt(args.input, args.gen_json)
    else:
        rtt_json(args.input, args.gen_json)

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- EDA ----------------------------------------------
#--------------------------------------------------------------------------------------------------

if args.method == "eda" and args.input != None:
    if args.gen_json == False:
        eda_txt(args.input, args.gen_json)
    else:
        eda_json(args.input, args.gen_json)

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- PARA ---------------------------------------------
#--------------------------------------------------------------------------------------------------

if args.method == "para" and args.input != None:
    if args.gen_json == False:
        para_txt(args.input, args.gen_json)
    else:
        para_json(args.input, args.gen_json)

#--------------------------------------------------------------------------------------------------
#----------------------------------------------- GPT ----------------------------------------------
#--------------------------------------------------------------------------------------------------

if args.method == "gpt" and args.input != None and args.amount != None and args.amount > 0:
    if args.gen_json == False:
        gpt_txt(input=args.input, gen_json=args.gen_json, amount=args.amount)
    else:
        gpt_json(input=args.input, gen_json=args.gen_json, amount=args.amount)

elif  args.method == "gpt" and args.input != None and args.amount == None:
    if args.gen_json == False:
        gpt_txt(input=args.input, gen_json=args.gen_json)
    else:
        gpt_json(input=args.input, gen_json=args.gen_json)

if args.method == "train_gpt" and args.input != None:
    train_gpt(args.input)