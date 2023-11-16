# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import optuna
import sys
import os
import argparse
import json
import gc
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from peft import PeftModel
from transformers import AutoModelForCausalLM

from utils import BioGPTLOTUSDataset, get_logger, evaluate

# Get arguments
parser = argparse.ArgumentParser()

parser.add_argument("--source-model", help="model of the source model from HF", type=str, required=True, dest='model_hf')
parser.add_argument("--lora-adapters", help="path to the LoRA adapters directoy", type=str, required=True, dest='lora_adapters')
parser.add_argument("--valid", help="path to json train file", type=str, required=True, dest='valid_path')
parser.add_argument("--tag", help="try tag", type=str, required=True, dest='tag')
parser.add_argument("--n-trials", help="Number of trial for the study", type=int, required=True, dest='n_trials')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')

args = parser.parse_args()

RELATION_VERBALISED="{org} produces {chem}"
MAX_LENGTH=1024
VALID_BATCH_SIZE = 4

lora_adapters = args.lora_adapters
study_tag = args.tag
n_trials = args.n_trials

output_dir = args.out_dir

valid_path = args.valid_path

model_hf = args.model_hf

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logger = get_logger("decoding-strat-hp-tuning" + study_tag, path=os.path.join(output_dir, "decoding-strat-hp-tuning-" + study_tag + ".log"))

if not os.path.exists(lora_adapters):
    logger.info("Can't find LoRA adapteres directoy")
    sys.exit(1)

# READ
with open(valid_path, "r") as f_input_valid:
  valid = json.load(f_input_valid)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_hf)

# Load validation set 
valid_dataset = BioGPTLOTUSDataset(data=valid, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle = True, num_workers = 0)

def objective(trial):

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_hf, device_map={"":0})
    model = PeftModel.from_pretrained(model, lora_adapters)
    model = model.merge_and_unload()
    model.to(device)

    EVAL_GENERATION_ARGS = {"max_new_tokens": 512, "do_sample": False, "forced_eos_token_id": tokenizer.eos_token_id}

    strat = trial.suggest_categorical("startegy", ["greedy", "beam-search"])

    if strat == "greedy":
        EVAL_GENERATION_ARGS["num_beams"] = 1

    else:
        nb_beams = trial.suggest_categorical("beam-size", [3, 5])
        stopping = trial.suggest_categorical("stoppingCriteria", [True, False, "never"])
        length_penalty = trial.suggest_float("length_penalty", 0, 3, step=0.5)
        EVAL_GENERATION_ARGS["num_beams"] = nb_beams
        EVAL_GENERATION_ARGS["early_stopping"] = stopping
        EVAL_GENERATION_ARGS["length_penalty"] = length_penalty
    
    # Just as a boolean, with fixed value to 10 if choosen
    bool_no_repeat_ngram_size = trial.suggest_categorical("bool_no_repeat_ngram_size", [True, False])
    if bool_no_repeat_ngram_size:
        EVAL_GENERATION_ARGS["no_repeat_ngram_size"] = 30

    logger.info("Generation config: %s", EVAL_GENERATION_ARGS)

    # Test on validation set:
    torch.cuda.empty_cache()
    eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("On eval - loss = %.3f - precision = %.3f - recall = %.3f - f1-score = %.3f", eval_metrics["loss"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1-score"])

    return eval_metrics["f1-score"]
    

study_name = "decoding-strat-hp-ft"
storage_name = f"sqlite:///{os.path.join(output_dir, study_name)}.db"

study = optuna.create_study(study_name=study_name, 
    storage=storage_name, 
    sampler=TPESampler(),
    direction="maximize", 
    load_if_exists=True)

study.optimize(objective, 
    n_trials=n_trials, 
    gc_after_trial=True,
    callbacks=[MaxTrialsCallback(40, states=None)])
