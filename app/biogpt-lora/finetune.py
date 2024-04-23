# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import argparse
import json
import io
import torch
import gc
import bitsandbytes
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
from statistics import mean
from contextlib import redirect_stdout

from utils import BioGPTLOTUSDataset, print_trainable_parameters, get_logger, evaluate

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model-name", help="model name to dl from HF repo", type=str, required=True, dest='model_hf')
parser.add_argument("--train", help="path to json train file", type=str, required=True, dest='train_path')
parser.add_argument("--valid", help="path to json train file", type=str, required=True, dest='valid_path')
parser.add_argument("--batch_size", help="batch size to use for the training set. Can be 4, 8 or 16. Gradient accumulation steps are automatically determined from the batch size", type=int, required=True, dest='batch_size', choices=[4, 6, 8, 12, 16])
parser.add_argument("--valid-batch_size", help="batch size to use for the validation set.", type=int, required=False, dest='valid_batch_size', default=8)
parser.add_argument("--r_lora", help="r (rank dimension) for lora", type=int, required=True, dest='r_lora')
parser.add_argument("--alpha_lora", help="alpha (scaling) for lora", type=int, required=True, dest='alpha_lora')
parser.add_argument("--lr", help="Learning rate. Warmup steps are fixed to 100.", type=float, required=True, dest='lr')
parser.add_argument("--num-epochs", help="Number of epochs.", type=int, required=False, dest='num_epoch', default=15)
parser.add_argument("--eval-steps", help="Compute evaluation loss every n steps", type=int, required=False, dest='eval_steps', default=20)
parser.add_argument("--log-steps", help="Save log metadata every n steps", type=int, required=False, dest='log_steps', default=1)
parser.add_argument("--n-epochs-before-eval", help="Number of epochs to processed before computing the first evaluation step", type=int, required=False, dest='epochs_bf_eval', default=2)
parser.add_argument("--n-warmup", help="Number of warmup steps", type=int, required=False, dest='n_warmup', default=100)
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')

args = parser.parse_args()

# VARS
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "output_projection"]
LORA_CONFIG = LoraConfig(
    r=args.r_lora,
    lora_alpha=args.alpha_lora,
    target_modules=TARGET_MODULES,
    lora_dropout=0.05,
    fan_in_fan_out=False,
    bias="none",
    task_type="CAUSAL_LM"
)

RELATION_VERBALISED="{org} produces {chem}"
MAX_LENGTH = 1024
NUM_EPOCHS = int(args.num_epoch)
TRAIN_BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.valid_batch_size
EVAL_STEP = int(args.eval_steps)
LOG_STEP = int(args.log_steps)
# FIX: set temperature to one for geedy decoding. 
EVAL_GENERATION_ARGS = {"do_sample": False, "num_beams": 1, "max_length": MAX_LENGTH, "temperature": 1}
MIN_EPOCH_BEFORE_EVAL = int(args.epochs_bf_eval)

GRADIENT_ACCUMULATION_STEP = 1
if TRAIN_BATCH_SIZE == 4:
    GRADIENT_ACCUMULATION_STEP = 20
elif TRAIN_BATCH_SIZE == 6:
    GRADIENT_ACCUMULATION_STEP = 13
elif TRAIN_BATCH_SIZE == 8:
    GRADIENT_ACCUMULATION_STEP = 10
elif TRAIN_BATCH_SIZE == 12:
    GRADIENT_ACCUMULATION_STEP = 7
else:
    GRADIENT_ACCUMULATION_STEP = 5

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
model = AutoModelForCausalLM.from_pretrained(model_hf, quantization_config=BNB_CONFIG, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(model_hf)
model.config.use_cache = False

EVAL_GENERATION_ARGS["forced_eos_token_id"] = tokenizer.eos_token_id

train_dataset = BioGPTLOTUSDataset(data=train, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)

valid_dataset = BioGPTLOTUSDataset(data=valid, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle = True, num_workers = 0)
valid_dataloader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle = True, num_workers = 0)

# KBIT CONVERT
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LORA_CONFIG)

optimizer_args = {"is_paged": True, "optim_bits": 8, "lr": args.lr}
optimizer = bitsandbytes.optim.AdamW(model.parameters(), **optimizer_args)

n_batches = (len(train_dataloader) * NUM_EPOCHS)
n_steps = n_batches // GRADIENT_ACCUMULATION_STEP + (1 if (n_batches % GRADIENT_ACCUMULATION_STEP) != 0 else 0)
n_warmup = int(args.n_warmup)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=n_warmup,
    num_training_steps=n_steps,
)

model.gradient_checkpointing_enable()
model.to(device)

logger = get_logger("qlora-trainer", path=os.path.join(output_dir, "biogpt-qlora-training.log"))
logger.info("Training batch-size: %d", TRAIN_BATCH_SIZE)
logger.info("Learning rate: %.2e", args.lr)
logger.info("Lora config: %s", LORA_CONFIG)
log_save = []
n_steps_per_epoch = (len(train_dataloader) // GRADIENT_ACCUMULATION_STEP) + (1 if (len(train_dataloader) % GRADIENT_ACCUMULATION_STEP != 0) else 0)
logger.info("%s steps per epoch", n_steps_per_epoch)
logger.info(print_trainable_parameters(model))


EVAL_F1 = 0

for epoch in range(NUM_EPOCHS):
  
    logger.info("Start epoch %d" %(epoch + 1))
    
    epoch_loss_l = []
    step = 0

    if (epoch + 1) > MIN_EPOCH_BEFORE_EVAL:
        eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
        logger.info("Eval - Epoch (%d / %d) - loss = %.3f - precision = %.3f - recall = %.3f - f1-score = %.3f", (epoch + 1), NUM_EPOCHS, eval_metrics["loss"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1-score"])
        log_save.append({"on": "eval", "epoch": (epoch + 1), "batch": -1, "loss": eval_metrics["loss"], "precision": eval_metrics["precision"], "recall": eval_metrics["recall"], "f1-score": eval_metrics["f1-score"]})
        
        #  save checkpoint ?
        if eval_metrics["f1-score"] > EVAL_F1:
            logger.info("Find new best checkpoint after %d epoch(s) with micro f1-score = %.2f", epoch, eval_metrics["f1-score"])
            EVAL_F1 = eval_metrics["f1-score"]
            # save
            model.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "best-model.json"), "w", encoding="utf-8") as f_best:
                json.dump({"epoch": epoch, "recall": eval_metrics["recall"] , "precision": eval_metrics["precision"], "f1.score": eval_metrics["f1-score"]}, f_best, indent=4)

    with redirect_stdout(io.StringIO()) as f:
        model.train()

    for index, batch in enumerate(train_dataloader):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss

        epoch_loss_l.append(loss.detach().cpu().item())

        loss = loss / GRADIENT_ACCUMULATION_STEP
        loss.backward()
        
        # Gradient accumulation
        if (index + 1) % GRADIENT_ACCUMULATION_STEP == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Is it a log step ?
            if step % LOG_STEP == 0:
                current_epoch_loss = mean(epoch_loss_l)
                logger.info("Train - Epoch (%d / %d) - Batch (%d / %d) - lr = %.2e - batch-loss = %.3f - epoch-loss = %.3f", (epoch+1), NUM_EPOCHS, (index + 1), len(train_dataloader), lr_scheduler.get_lr()[0], epoch_loss_l[-1], current_epoch_loss)
                log_save.append({"on": "train", "epoch": (epoch + 1), "batch": (index + 1), "lr": lr_scheduler.get_lr()[0], "batch-loss": epoch_loss_l[-1], "epoch-loss": current_epoch_loss})
            
            # Only eval with generation on beginning of epoch
            if step % EVAL_STEP == 0:
                gc.collect()
                torch.cuda.empty_cache()
                eval_metrics = evaluate(model, valid_dataloader, tokenizer, False, EVAL_GENERATION_ARGS, device, logger)
                logger.info("Eval - Epoch (%d / %d) - Batch (%d / %d) - loss = %.3f", (epoch + 1), NUM_EPOCHS, (index + 1), len(train_dataloader), eval_metrics["loss"])
                log_save.append({"on": "eval", "epoch": (epoch + 1), "batch": (index + 1), "loss": eval_metrics["loss"]})
                with redirect_stdout(io.StringIO()) as f:
                    model.train()
                torch.cuda.empty_cache()
    
    # if the last pass was not an update
    if (len(train_dataloader) % GRADIENT_ACCUMULATION_STEP) != 0:
        logger.info("Last update of the epoch.")
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()


# Last eval:
eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
logger.info("Eval - Final eval - loss = %.3f - precision = %.3f - recall = %.3f - f1-score = %.3f", eval_metrics["loss"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1-score"])
log_save.append({"on": "eval", "epoch": NUM_EPOCHS, "batch": -1, "loss": eval_metrics["loss"], "precision": eval_metrics["precision"], "recall": eval_metrics["recall"], "f1-score": eval_metrics["f1-score"]})

# save log
with open(os.path.join(output_dir, "log-stats.json"), "w", encoding="utf-8") as f_log_stats:
    json.dump(log_save, f_log_stats, indent=4)

#  save final checkpoint ?
if eval_metrics["f1-score"] > EVAL_F1:
    logger.info("Find new best checkpoint after LAST epoch with micro f1-score = %.2f", eval_metrics["f1-score"])
    # save
    model.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "best-model.json"), "w", encoding="utf-8") as f_best:
        json.dump({"epoch": NUM_EPOCHS, "recall": eval_metrics["recall"] , "precision": eval_metrics["precision"], "f1.score": eval_metrics["f1-score"]}, f_best, indent=4)

logger.info("End.")