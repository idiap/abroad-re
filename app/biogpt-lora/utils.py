# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import pandas as pd
import torch
import numpy as np
import io
import logging
from torch.utils.data import Dataset
import re
import bitsandbytes
import logging
from collections import defaultdict
from contextlib import redirect_stdout
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, AutoModelForCausalLM


def batch_predict(model, batch, tokenizer, generate_args, device, logger):

    BOS_TOKEN_ID = tokenizer.bos_token_id
    PAD_TOKEN_ID = tokenizer.pad_token_id
    EOS_TOKEN_ID = tokenizer.eos_token_id

    # Extract input ids and prepare for prompting with only the input abstract: 
    generation_inputs = batch["input_ids"].clone()
    if tokenizer.name_or_path == "BioMistral/BioMistral-7B":
        starts_generation = (generation_inputs[:, 1:] == BOS_TOKEN_ID).nonzero()[:, 1] + 1
    else:
        starts_generation = (generation_inputs == BOS_TOKEN_ID).nonzero()[:, 1]
    padding_length = starts_generation.max()
    generation_inputs = generation_inputs[:, :(padding_length + 1)]
    generation_attemtion_mask = batch["attention_mask"].clone()[:, :(padding_length + 1)]

    # padd to the left
    for i in range(generation_inputs.shape[0]):
        left_padded_input = torch.tensor([PAD_TOKEN_ID] * (padding_length + 1))
        left_padded_attention = torch.zeros((padding_length + 1))
        left_padded_input[(padding_length - starts_generation[i]):] = generation_inputs[i][:(starts_generation[i] + 1)]
        left_padded_attention[(padding_length - starts_generation[i]):] = 1
        generation_inputs[i] = left_padded_input
        generation_attemtion_mask[i] = left_padded_attention
    
    generation = model.generate(input_ids=generation_inputs.to(device), attention_mask=generation_attemtion_mask.to(device), **generate_args)
    
    decoded_output = []
    for j in range(generation.shape[0]):
        gen = generation[j].detach().cpu()[(padding_length + 1): ]
        ends_at = (gen == EOS_TOKEN_ID).nonzero()[-1]
        decoded_output.append(tokenizer.decode(gen[:ends_at]))
    
    return decoded_output


def parse_predictions(prediction_str, separator, template_sep, logger):
    
    def remove_articles(text):
        regex = re.compile(r"\b(an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return "".join(text.split())
    
    relations = defaultdict(lambda: [])
    items = prediction_str.strip().split(separator)

    if not len(items):
       logger.debug("Output predictions '%s', cannot be parsed with separator %s", prediction_str, separator)
       return relations

    for item in items:
        try:
           org, chem = item.split(template_sep)
        except ValueError as e_v:
           logger.debug("Fail during parsing of output prediction: %s.", item)
           continue

        org = white_space_fix(remove_articles(org.strip(' .').lower()))
        chem = white_space_fix(remove_articles(chem.strip(' .').lower()))

        if chem not in relations[org]:
            relations[org].append(chem)
    
    return relations

def prepare_decoded_labels(batch_labels, tokenizer):
    
    recovered_labels = []
    for b in batch_labels:
       _indexes = (b != -100).nonzero()
       # Don't consider the </s> at the end
       recovered_labels.append(tokenizer.decode(b[_indexes[0]:(_indexes[-1])]))
    
    return recovered_labels

def compute_metrics(preds, labels, logger):

    SEP = "; "
    TEMPLATE = "produces"

    TOTAL_P = 0
    TOTAL_TP = 0
    TOTAL_FP = 0

    # Checks dims
    assert len(preds) == len(labels)
    batch_len = len(preds)

    for i in range(batch_len):

        parsed_preds = parse_predictions(preds[i], SEP, TEMPLATE, logger)
        parsed_labels = parse_predictions(labels[i], SEP, TEMPLATE, logger)

        TOTAL_P += sum([len(l) for l in parsed_labels.values()])

        if not len(parsed_preds):
            logger.debug("Empty predictions, skip")
            continue

        FP = 0
        TP = 0

        for pred_org, pred_chem_list in parsed_preds.items():
            for pred_chem in pred_chem_list:
                
                # Check if it is a TP
                if pred_chem in parsed_labels[pred_org]:
                    TP += 1
                else:
                    FP += 1

        # Add to the total counter
        TOTAL_TP += TP
        TOTAL_FP += FP

    return TOTAL_P, TOTAL_TP, TOTAL_FP

def evaluate(model, validation_dataloader, tokenizer, do_generate_eval, generate_args, device, logger):
  
    logger.info("Start Evaluation")

    total_loss = 0
    total_positives = 0
    total_true_positives = 0
    total_false_positives = 0
    micro_f1 = 0
    micro_precision = 0
    micro_recall = 0

    SEP = "; "
    TEMPLATE = "produces"
    with redirect_stdout(io.StringIO()) as f:
        model.eval()

    n = len(validation_dataloader)

    with torch.no_grad():
        for index, batch in enumerate(validation_dataloader):

            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = output.loss.detach().cpu().item()
            logger.info("In Eval - Batch (%d / %d) - loss: %.3f", (index + 1), len(validation_dataloader), loss)
            total_loss += loss
        
            if do_generate_eval:
               logger.info("In Eval - compute metrics for batch (%d / %d)", (index + 1), len(validation_dataloader))
               decoded_output = batch_predict(model, batch, tokenizer, generate_args, device, logger)
               recovered_labels = prepare_decoded_labels(batch["labels"].detach().cpu(), tokenizer)
               print(decoded_output)
               print(recovered_labels)
               batch_total_P, batch_total_TP, batch_total_FP = compute_metrics(decoded_output, recovered_labels, logger)

               total_positives += batch_total_P
               total_true_positives += batch_total_TP
               total_false_positives += batch_total_FP

    if total_true_positives != 0 or total_false_positives != 0:
        micro_precision = total_true_positives / (total_true_positives + total_false_positives)

    if total_positives != 0:
        micro_recall = total_true_positives / total_positives

    if micro_precision != 0 or micro_recall != 0:
        micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))

    total_loss = total_loss / n

    if do_generate_eval:
        return {"loss": total_loss, "precision": micro_precision, "recall": micro_recall, "f1-score": micro_f1}
    else:
        logger.info("Return only global loss")
        return {"loss": total_loss}


class BioGPTLOTUSDataset(Dataset):
  """
  Truncate input text if too long
  """

  def __init__(self, data, tokenizer, template, relation_separator, max_length):
    self.data = list(data.values())
    self.tokenizer = tokenizer
    self.template = template
    self.relation_separator = relation_separator
    self.max_length = max_length
    self.BOS = self.tokenizer.bos_token
    self.BOS_id = self.tokenizer.bos_token_id
    self.EOS = self.tokenizer.eos_token
    self.EOS_id = self.tokenizer.eos_token_id

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):

    # get labels
    chems = pd.DataFrame(self.data[index]["chemicals"])
    orgs = pd.DataFrame(self.data[index]["organisms"])

    # get input_text tokens:
    sep = self.relation_separator
    input_text = self.data[index]["ArticleTitle"].strip("\n ") + " " + self.data[index]["AbstractText"].strip("\n ")
    raw_output_text = sep.join([self.template.format(org=orgs[orgs["id"] == org_id]["label"].values[0], chem=chems[chems["id"] == chem_id]["label"].values[0]) for (org_id, chem_id) in self.data[index]["relations"]])

    input_text = input_text
    output_text = self.EOS + self.BOS + raw_output_text + self.EOS

    # merge and toknenize
    global_input = input_text + output_text
    tokenized = self.tokenizer(global_input, max_length=self.max_length, padding='max_length', truncation=True)
    # print(tokenized)

    # Was the content truncated ?
    if tokenized["input_ids"][-1] != self.tokenizer.pad_token_id:
        # get the length of the tokenized output. We suppose that it is at least always less than the context window
        # Warning: In the case of bioGpt the tokenizer auto-happen an eos at the beginining, while in the case of Mistral it's a BOS 
        if self.tokenizer.name_or_path == "BioMistral/BioMistral-7B":
            new_output_text = raw_output_text + self.EOS
        else:
            new_output_text = self.BOS + raw_output_text + self.EOS
        
        tokenized_output_text = self.tokenizer(new_output_text, max_length=self.max_length, padding='max_length', truncation=True)

        # Get the tokenized input text. Likely this was too long.
        tokenized_input_text = self.tokenizer(input_text, max_length=self.max_length, padding='max_length', truncation=True)["input_ids"]
        # print(tokenized_input_text)

        # if we want to keep all the output, then this is the remaining space we have to put all the input tokens
        length_output = np.count_nonzero(tokenized_output_text["attention_mask"])
        remaining_after_output = self.max_length - length_output
        
        # As we will have to add manually the EOS for biomistral
        if self.tokenizer.name_or_path == "BioMistral/BioMistral-7B":
            remaining_after_output = remaining_after_output - 1

        # The input in truncated to this size
        truncated_tokenized_input_text = tokenized_input_text[:remaining_after_output]
        assert len(truncated_tokenized_input_text) == remaining_after_output

        # remerge everything together
        if self.tokenizer.name_or_path == "BioMistral/BioMistral-7B":
            merged_tokenized_input_output = truncated_tokenized_input_text + [self.EOS_id] + tokenized_output_text["input_ids"][:length_output]
        else:
            merged_tokenized_input_output = truncated_tokenized_input_text + tokenized_output_text["input_ids"][:length_output]
        assert len(merged_tokenized_input_output) == self.max_length

        # The attention mask is irrelevant if we reach the end of the context windowas there is no padding left.
        merged_attention_mask = [1] * self.max_length
        tokenized["input_ids"] = merged_tokenized_input_output
        tokenized["attention_mask"] = merged_attention_mask

    # get the start and end of the output (the labels)
    if self.tokenizer.name_or_path == "BioMistral/BioMistral-7B":
        # Skip the first BOS token for Biomistral
        start_of_output = tokenized["input_ids"][1:].index(self.BOS_id) + 1
    else:
        # If not do as for BioGPT
        start_of_output = tokenized["input_ids"].index(self.BOS_id)

    end_of_output = len(tokenized["input_ids"]) - tokenized["input_ids"][::-1].index(self.EOS_id) - 1
    # Create the output labels.
    labels = tokenized["input_ids"].copy()

    # We do not consider any of the input_text tokens, the BOR token and the PAD tokens in the loss
    labels[:(start_of_output + 1)] = [-100] * (start_of_output + 1)
    labels[(end_of_output + 1):] = [-100] * (self.max_length - end_of_output - 1)

    # Checking
    assert len(tokenized["input_ids"]) == len(tokenized["attention_mask"]) == len(labels)

    tokenized["input_ids"] = torch.tensor(tokenized["input_ids"])
    tokenized["attention_mask"] = torch.tensor(tokenized["attention_mask"])
    tokenized["labels"] = torch.tensor(labels)
    
    return tokenized

def get_logger(name, **kwargs):
    """Create a default logger

    Returns:
        logging.logger: a default logger
    """

    # set loggers
    level = kwargs.get("level", logging.INFO)
    log_path = kwargs.get("path", "./training.log")
    open(log_path, 'w', encoding="utf-8").close()

    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if kwargs.get("stdout", False):
        log_file_handler = logging.FileHandler(filename=log_path)
        log_file_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_file_handler)
        log_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        log_stdout_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_stdout_handler)
    else:
        log_file_handler = logging.FileHandler(filename=log_path)
        log_file_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_file_handler)

    return logger


def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params, all_param = model.get_nb_trainable_parameters()
  
  return f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"


def define_context(model_hf, train_dataset, r_lora, alpha_lora, batch_size, lr, gradient_accumulation_step, num_epochs):

    # Dataloader with batchsize
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers = 0)

    # Model init
    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_hf, quantization_config=BNB_CONFIG, device_map={"":0})
    model.config.use_cache = False

    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "output_projection"]

    # Prepare LORA conf
    lora_config = LoraConfig(
        r=r_lora,
        lora_alpha=alpha_lora,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        fan_in_fan_out=False,
        bias="none",
        task_type="CAUSAL_LM")

    # KBIT CONVERT
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # optimizer
    optimizer_args = {"is_paged": True, "optim_bits": 8, "lr": lr}
    optimizer = bitsandbytes.optim.AdamW(model.parameters(), **optimizer_args)

    # Prepare scheduler
    n_batches = (len(train_dataloader) * num_epochs)
    n_steps = n_batches // gradient_accumulation_step + (1 if (n_batches % gradient_accumulation_step) != 0 else 0)
    n_warmup = 100
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=n_warmup,
        num_training_steps=n_steps)

    return model, optimizer, train_dataloader, lr_scheduler
