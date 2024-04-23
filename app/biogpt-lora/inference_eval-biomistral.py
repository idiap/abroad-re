# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import os
import argparse
import json
import gc
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils import BioGPTLOTUSDataset, get_logger, evaluate

# Get arguments
parser = argparse.ArgumentParser()

parser.add_argument("--source-model", help="model of the source model from HF", type=str, required=True, dest='model_hf')
parser.add_argument("--merged-model", help="path to the LoRA adapters directoy", type=str, required=True, dest='merged_model')
parser.add_argument("--test", help="path to json train file", type=str, required=True, dest='test_path')
parser.add_argument("--hf", help="Look for adapters on Hugging Face", required=False, default=False, action='store_true', dest="adapters_on_hf")
parser.add_argument("--output-dir", help="path to the output dir", type=str, required=True, dest='out_dir')
parser.add_argument("--valid-b-size", help="path to json train file", type=int, required=False, dest='valid_batch_size', default=4)

args = parser.parse_args()

TOKENIZER_CONFIG = {
"padding_side": "right",
"pad_token": '<unk>'
}

MAX_LENGTH = 2048

RELATION_VERBALISED="{org} produces {chem}"

VALID_BATCH_SIZE = args.valid_batch_size

merged_model = args.merged_model

output_dir = args.out_dir

test_path = args.test_path

model_hf = args.model_hf

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


name = os.path.basename(args.merged_model)

logger = get_logger("eval-biogpt-qlora-model", path=os.path.join(output_dir, name + "-eval-biogpt-qlora-model.log"))

if not args.adapters_on_hf and not os.path.exists(merged_model):
    logger.info("Can't find merged model directoy")
    sys.exit(1)

# READ
with open(test_path, "r") as f_input_test:
    test = json.load(f_input_test)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_hf, **TOKENIZER_CONFIG)

EVAL_GENERATION_ARGS = {"do_sample": False, 
    "num_beams": 3,
    "length_penalty": 3,
    "max_length": MAX_LENGTH,
    "temperature": 1,
    "forced_eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id}

# Load validation set 
test_dataset = BioGPTLOTUSDataset(data=test, tokenizer=tokenizer, template=RELATION_VERBALISED, relation_separator="; ", max_length=MAX_LENGTH)
test_dataloader = DataLoader(test_dataset, batch_size=VALID_BATCH_SIZE, shuffle = True, num_workers = 0)

# Load model
BNB_CONFIG = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(merged_model, quantization_config=BNB_CONFIG, device_map={"":0})
print(model)
logger.info("Generation config: %s", EVAL_GENERATION_ARGS)

# Test on validation set:
torch.cuda.empty_cache()
eval_metrics = evaluate(model, test_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
torch.cuda.empty_cache()
gc.collect()
logger.info("On eval - loss = %.3f - precision = %.3f - recall = %.3f - f1-score = %.3f", eval_metrics["loss"], eval_metrics["precision"], eval_metrics["recall"], eval_metrics["f1-score"])
