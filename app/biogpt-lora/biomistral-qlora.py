# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import json
import io
import torch
import gc
import wandb
import hydra
import bitsandbytes

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup
from statistics import mean
from contextlib import redirect_stdout

from utils import BioGPTLOTUSDataset, print_trainable_parameters, get_logger, evaluate

# Run as test: python app/biogpt-lora/biomistral-qlora.py wandb.name=test training/data=mixtral_1000

# wandb login
wandb.login()

@hydra.main(version_base=None, config_path="../../config", config_name="training_biomistral")
def run(config: DictConfig) -> None:
    
    OUTPUT_DIR = config.training.output_dir
    GAS = config.training.model.gradient_accumulation_step
    NUM_EPOCHS = config.training.model.num_epochs

    # Set the project where this run will be logged and track hyperparameters and run metadata
    wandb.init(**config.wandb, config=config.training)

    # wanfb metrics
    wandb.define_metric("epoch_step", step_sync=False, hidden=True)
    wandb.define_metric("epoch_precision", step_metric="epoch_step")
    wandb.define_metric("epoch_recall", step_metric="epoch_step")
    wandb.define_metric("epoch_f1_score", step_metric="epoch_step")
    wandb.define_metric("epoch_validation_loss", step_metric="epoch_step")

    wandb.define_metric("eval_steps", step_sync=False, hidden=True)
    wandb.define_metric("validation_loss", step_metric="eval_steps")

    device = torch.device("cuda")

    BNB_CONFIG = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    TOKENIZER_CONFIG = {
    "padding_side": "right",
    "pad_token": '<unk>'
    }
    LORA_CONFIG = LoraConfig(
        r=config.training.model.r_lora,
        lora_alpha=config.training.model.alpla_lora,
        target_modules=list(config.training.model.target_module),
        lora_dropout=0.05,
        fan_in_fan_out=False,
        bias="none",
        task_type="CAUSAL_LM")

    # Read data
    with open(config.training.data.train_path, "r") as f_input_train:
        train = json.load(f_input_train)
    with open(config.training.data.valid_path, "r") as f_input_valid:
        valid = json.load(f_input_valid)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.training.model.model_hf, quantization_config=BNB_CONFIG, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(config.training.model.model_hf, **TOKENIZER_CONFIG)

    model.config.use_cache = False

    EVAL_GENERATION_ARGS = {"do_sample": False, 
        "num_beams": 1,
        "max_length": config.training.model.max_length,
        "temperature": 1,
        "forced_eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id}

    train_dataset = BioGPTLOTUSDataset(data=train, 
        tokenizer=tokenizer, 
        template=config.training.model.relation_verbalised, 
        relation_separator="; ", 
        max_length=config.training.model.max_length)

    valid_dataset = BioGPTLOTUSDataset(data=valid, 
        tokenizer=tokenizer,
        template=config.training.model.relation_verbalised,
        relation_separator="; ",
        max_length=config.training.model.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.model.train_batch_size, shuffle = True, num_workers = 0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.training.model.valid_batch_size, shuffle = True, num_workers = 0)

    # KBIT CONVERT
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LORA_CONFIG)

    optimizer_args = {"is_paged": True, "optim_bits": 8, "lr": config.training.model.lr}
    optimizer = bitsandbytes.optim.AdamW(model.parameters(), **optimizer_args)

    n_batches = (len(train_dataloader) * NUM_EPOCHS)
    n_steps = n_batches // GAS + (1 if (n_batches % GAS) != 0 else 0)

    lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.training.model.n_warmup,
    num_training_steps=n_steps,
    )

    model.gradient_checkpointing_enable()
    model.to(device)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    logger = get_logger("biomistral-qlora-trainer", path=os.path.join(OUTPUT_DIR, "biomistral-qlora-training.log"), stdout=True)
    logger.info(print_trainable_parameters(model))

    EVAL_F1 = 0
    EVAL_STEP = 1
    for epoch in range(NUM_EPOCHS):
        logger.info("Start epoch %d" %(epoch + 1))
        step = 0

        if (epoch + 1) > config.training.model.min_epochs_before_eval:
            eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
            wandb.log({"epoch_step": (epoch + 1), "epoch_validation_loss": eval_metrics["loss"], "epoch_precision": eval_metrics["precision"], "epoch_recall": eval_metrics["recall"], "epoch_f1_score": eval_metrics["f1-score"]})
            
            # save checkpoint ?
            if eval_metrics["f1-score"] > EVAL_F1:
                logger.info("Find new best checkpoint after %d epoch(s) with micro f1-score = %.2f", epoch, eval_metrics["f1-score"])
                EVAL_F1 = eval_metrics["f1-score"]
                
                # save
                model.save_pretrained(OUTPUT_DIR)
                with open(os.path.join(OUTPUT_DIR, "best-model.json"), "w", encoding="utf-8") as f_best:
                    json.dump({"epoch": epoch, "recall": eval_metrics["recall"] , "precision": eval_metrics["precision"], "f1.score": eval_metrics["f1-score"]}, f_best, indent=4)
        
        # training loop
        with redirect_stdout(io.StringIO()) as f:
                model.train()
        
        for index, batch in enumerate(train_dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            loss = loss / GAS
            loss.backward()
            
            # Gradient accumulation
            if (index + 1) % GAS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1
                wandb.log({"loss": loss})
                logger.info("Train - Epoch (%d / %d) - Batch (%d / %d) - lr = %.2e", (epoch+1), NUM_EPOCHS, (index + 1), len(train_dataloader), lr_scheduler.get_lr()[0])
                 
                # Only eval with generation on beginning of epoch
                if step % config.training.model.eval_step == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    eval_metrics = evaluate(model, valid_dataloader, tokenizer, False, EVAL_GENERATION_ARGS, device, logger)
                    wandb.log({"eval_steps": EVAL_STEP, "validation_loss": eval_metrics["loss"]})
                    logger.info("Eval - Epoch (%d / %d) - Batch (%d / %d)", (epoch + 1), NUM_EPOCHS, (index + 1), len(train_dataloader))
                    EVAL_STEP += 1
                    with redirect_stdout(io.StringIO()) as f:
                        model.train()
                    torch.cuda.empty_cache()
        
        # if the last pass was not an update
        if (len(train_dataloader) % GAS) != 0:
            logger.info("Last update of the epoch.")
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    
    # Last eval:
    eval_metrics = evaluate(model, valid_dataloader, tokenizer, True, EVAL_GENERATION_ARGS, device, logger)
    wandb.log({"epoch_step": (epoch + 1), "epoch_validation_loss": eval_metrics["loss"], "epoch_precision": eval_metrics["precision"], "epoch_recall": eval_metrics["recall"], "epoch_f1_score": eval_metrics["f1-score"]})

    # save final checkpoint ?
    if eval_metrics["f1-score"] > EVAL_F1:
        logger.info("Find new best checkpoint after LAST epoch with micro f1-score = %.2f", eval_metrics["f1-score"])
        
        # save
        model.save_pretrained(OUTPUT_DIR, save_embedding_layers=False)
        with open(os.path.join(OUTPUT_DIR, "best-model.json"), "w", encoding="utf-8") as f_best:
            json.dump({"epoch": NUM_EPOCHS, "recall": eval_metrics["recall"] , "precision": eval_metrics["precision"], "f1.score": eval_metrics["f1-score"]}, f_best, indent=4)

    logger.info("End.")
    wandb.finish()


if __name__ == "__main__":
    run()