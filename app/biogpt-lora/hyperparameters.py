# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import optuna
import os
import argparse
import json
import torch
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from statistics import mean

from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback

from utils import BioGPTLOTUSDataset, print_trainable_parameters, get_logger, evaluate, define_context

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", help="model name to dl from HF repo", type=str, required=True, dest='model_hf')
parser.add_argument("--train", help="path to json train file", type=str, required=True, dest='train_path')
parser.add_argument("--valid", help="path to json train file", type=str, required=True, dest='valid_path')
parser.add_argument("--tag", help="try tag", type=str, required=True, dest='tag')
parser.add_argument("--n-trials", help="Number of trial for the study", type=int, required=True, dest='n_trials')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')

args = parser.parse_args()

# VARS
RELATION_VERBALISED="{org} produces {chem}"
MAX_LENGTH=1024
NUM_EPOCHS = 5
VALID_BATCH_SIZE = 8
EVAL_GENERATION_ARGS = {"max_new_tokens": 512, "early_stopping": True, "no_repeat_ngram_size": 30}
MIN_EPOCH_BEFORE_EVAL = 2

study_tag = args.tag
n_trials = args.n_trials

output_dir = args.out_dir

train_path = args.train_path

valid_path = args.valid_path

model_hf = args.model_hf

# READ
with open(train_path, "r") as f_input_train:
    train = json.load(f_input_train)
with open(valid_path, "r") as f_input_valid:
    valid = json.load(f_input_valid)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# LOAD
tokenizer = AutoTokenizer.from_pretrained(model_hf)
EVAL_GENERATION_ARGS["forced_eos_token_id"] = tokenizer.eos_token_id

# Load and split valid 
valid_dataset = BioGPTLOTUSDataset(data=valid, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle = True, num_workers = 0)

train_dataset = BioGPTLOTUSDataset(data=train, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)

logger = get_logger("qlora-trainer-hp-tuning" + study_tag, path=os.path.join(output_dir, "biogpt-qlora-hp-tuning" + study_tag + ".log"))

def objective(trial):
    
    # Same interval for biogpt and biogpt-large
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True) # Down to 3 order of magnitude

    # Scale the Gradient accumulation step by the batch size to guaranty the sane nb of steps.
    if model_hf == "microsoft/biogpt":
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    else:
        batch_size = trial.suggest_categorical("batch_size", [6, 12])

    gradient_accumulation_step = 1

    if batch_size == 4:
        gradient_accumulation_step = 20
    elif batch_size == 6:
        gradient_accumulation_step = 13
    elif batch_size == 8:
        gradient_accumulation_step = 10
    elif batch_size == 12:
        gradient_accumulation_step = 7
    else:
        gradient_accumulation_step = 5
    
    if model_hf == "microsoft/biogpt":
        LORA_PARAMETERS = [(4, 4), (4, 8), (4, 16), (8, 8), (8, 16), (8, 32), (16, 16), (16, 32), (16, 64)]
        lora_config = trial.suggest_categorical("lora_config", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    else:
        LORA_PARAMETERS = [(4, 4), (4, 8), (4, 16), (8, 8), (8, 16), (8, 32)]
        lora_config = trial.suggest_categorical("lora_config", [0, 1, 2, 3, 4, 5])

    r_lora = LORA_PARAMETERS[lora_config][0]
    alpha_lora = LORA_PARAMETERS[lora_config][1]

    logger.info("Trial:\n - lr = %.2e\n - batch = %d\n - GA = %d\n - r_lora = %d\n - alpha_lora = %d", lr, batch_size, gradient_accumulation_step, r_lora, alpha_lora)    

    model, optimizer, train_dataloader, lr_scheduler = define_context(model_hf, train_dataset, r_lora, alpha_lora, batch_size, lr, gradient_accumulation_step, NUM_EPOCHS)
    logger.info(print_trainable_parameters(model))
    model.to(device)
    model.train()

    for epoch in range(NUM_EPOCHS):

        epoch_loss_l = []

        logger.info("Epoch %d", (epoch + 1))

        model.train()

        for index, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            epoch_loss_l.append(loss.detach().cpu().item())

            loss = loss / gradient_accumulation_step
            loss.backward()
            
            # Gradient accumulation
            if (index + 1) % gradient_accumulation_step == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                logger.info("Train - Epoch (%d / %d) - Batch (%d / %d) - lr = %.2e - batch-loss = %.3f", (epoch+1), NUM_EPOCHS, (index + 1), len(train_dataloader), lr_scheduler.get_lr()[0], epoch_loss_l[-1])

                # prevent exploding loss for too high lr
                if epoch_loss_l[-1] > 10:
                    logger.info("Exploding loss, prune this step.")
                    return 0

        # Last batchs are underscale by gradient accumulation.
        if (len(train_dataloader) % gradient_accumulation_step) != 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        logger.info("Train - Epoch (%d / %d) - lr = %.2e - epoch-loss = %.3f", (epoch+1), NUM_EPOCHS, lr_scheduler.get_lr()[0], mean(epoch_loss_l))
    
        # Only evaluate from 2 epoch. Pruners also set to evaluate after 2 epochs
        if (epoch + 1) > MIN_EPOCH_BEFORE_EVAL:
            # Clena before eval
            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
            f1 = eval_metrics["f1-score"]
            torch.cuda.empty_cache()

            # We report the step so that it is the step=1 and we set n_warmup_steps=0. To goal is to start pruning possibly after the MIN_EPOCH_BEFORE_EVAL 
            trial.report(f1, (epoch - MIN_EPOCH_BEFORE_EVAL + 1))
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    logger.info("end")
    return f1

# study_name = args.study_name
study_name = "test-biogpt-Large-hp-ft"
storage_name = f"sqlite:///{os.path.join(output_dir, study_name)}.db"

study = optuna.create_study(study_name=study_name, 
    storage=storage_name, 
    sampler=TPESampler(),
    direction="maximize", 
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_startup_trials=40, n_warmup_steps=0))

study.optimize(objective, 
    n_trials=n_trials, 
    gc_after_trial=True,
    callbacks=[MaxTrialsCallback(80, states=None)])

# MaxTrialsCallback must be initialize to n_workers * n_trials !