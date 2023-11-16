# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate synthetic abstract"""
import os
import sys
import argparse
import json
import numpy as np
from tqdm import tqdm
from llama_cpp import Llama

from abstract_generator import LLamaAbstractGenerator, LotusSyntheticData, LotusAbstractSelector

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to the model file", type=str, required=True, dest='model_path')
parser.add_argument("--input-file", help="path to the data test file", type=str, required=True, dest='data')
parser.add_argument("-N", help="number of sampled abstract per PMID. Default 1.", type=int, required=False, dest='N')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')
parser.add_argument("--seed", help="seed", type=int, required=False, dest='seed', default=1024)
parser.add_argument("--m-threads", help="Number of threads use for the model. If all the layers are put on GPU, it may be faster to set the number of threads to 1.", type=int, required=False, default=1, dest="n_threads")
parser.add_argument("--m-n-gpu", help="Number of layers of the model put on GPUs. Ps: There are 43 layers in the Vicuna-13B", type=int, required=False, default=43, dest="n_gpu")
parser.add_argument("--score-threshold", help="the threshold on the score of an abstract below which it is discarded.", type=float, required=False, dest='score_th', default=0.9)

args = parser.parse_args()

# set model and generation params for llama.cpp
MODEL_PARAMS = {"n_ctx": 2048, "n_batch": 512, "n_threads": args.n_threads, "n_gpu_layers": args.n_gpu, "seed": args.seed}
GENERATION_PARAMS = {"max_tokens": 512, "stop": ["\n\n", "Instructions:", "Title:", "Keywords:", "Main findings:", "Abstract:"], "temperature": 0.7}
TEMPERATURE_RANGE = np.arange(0, 1.1, 0.1).tolist()
FINAL_DICT = {}
EXCLUDED_LIST = []
SCORE_TH = args.score_th

f_input_name = os.path.splitext(os.path.basename(args.data))[0]

generator = LLamaAbstractGenerator(model_path=args.model_path, model_params=MODEL_PARAMS, log_file_name="abstract-generator-" + f_input_name + ".log")
logger = generator.logger
n = args.N

selector = LotusAbstractSelector(logger=logger)

f_out_name = "out_" + f_input_name + ".json"
f_excluded_name = "out_" + f_input_name +  "_excluded.txt"

with open(args.data) as f_prompts:
    data_to_prompt = json.load(f_prompts)


# For all PMID items
for pmid, pmid_item in tqdm(data_to_prompt.items()):

    title = pmid_item["ArticleTitle"]

    synthetic_data = []

    logger.info("Run for PMID %s", pmid) 

    # For all generated prompts:
    for index, prompt_item in enumerate(pmid_item["prompts"]):

        prompt = prompt_item["prompt"]

        # Sample a temperature and generates
        temperature = np.random.choice(a=TEMPERATURE_RANGE, size=1, p=[1/len(TEMPERATURE_RANGE)] * len(TEMPERATURE_RANGE))[0]
        GENERATION_PARAMS["temperature"] = temperature

        # Generate an abstract
        logger.info("prompt %d: use temperature: %.2f", (index + 1), GENERATION_PARAMS["temperature"])
        generated_abstract = generator.generate_abstract(prompt=prompt, parameters=GENERATION_PARAMS)
        
        if generated_abstract:
            new_data = LotusSyntheticData(id = pmid + '-' + str(index), pmid=pmid, prompt=prompt, title=title, abstract=generated_abstract, chemicals=prompt_item["chemicals"], organisms=prompt_item["organisms"], relations=prompt_item["relations"], generation_params=GENERATION_PARAMS.copy())
            new_data.post_processing()
            synthetic_data.append(new_data)
        
    selected_synthetic_data = selector.select(list_of_synthetic_data=synthetic_data, n=n, score_th=SCORE_TH)

    if not len(selected_synthetic_data):
        logger.warning("No valid abstracts generated for PMID %s. Add to the exlcuded list.", pmid)
        EXCLUDED_LIST.append(pmid)

    # Browse and add:
    for s_data in selected_synthetic_data:
        FINAL_DICT[s_data.id] = s_data.export()

logger.info("Export generated abstracts")

with open(os.path.join(args.out_dir, f_out_name), "w", encoding="utf-8") as f_out_abstracts:
    json.dump(FINAL_DICT, f_out_abstracts, indent=4)

logger.info("Export excluded PMID list")
with open(os.path.join(args.out_dir, f_excluded_name), "w", encoding="utf-8") as f_out_excluded:
    for pmid in EXCLUDED_LIST:
        f_out_excluded.write(f"{pmid}\n")
logger.info("End.")