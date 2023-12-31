# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import argparse
from llama_cpp import Llama
import json
import pandas as pd

from urllib3.util import Retry
from urllib3 import PoolManager
from tqdm import tqdm
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from keyword_helpers import get_exclusion_list
from instructions_builder import LotusInstructionsBuilder


parser = argparse.ArgumentParser()
parser.add_argument("--input-file", help="path to the input file", type=str, required=True, dest='data')
parser.add_argument("--chemical-classes", help="path to the file containing the chemical classes annotations", type=str, required=False, dest='chemical_classes', default=None)
parser.add_argument("--method", help="The method use to extract the keywords. Can be either KeyBERT or prompting a LLM with llama.cpp", type=str, choices=["KeyBERT", "llama.cpp"], required=True, dest='method')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')
parser.add_argument("--cache-dir", help="path to the cache directory", type=str, required=False, dest='cache_dir', default=".cache")
parser.add_argument("--model-path-or-name", help="Sentence-transformers model name from HF for KeyBERT and path to the ggml file for llama.cpp", type=str, required=True, dest="model_path")
parser.add_argument("--use-pubtator", help="Use PubTator to extend the list of potential chemical or organism names from the original abstract", required=False, default=False, action='store_true', dest="use_pubtator")
parser.add_argument("--path-to-synonyms", help="Path to a tabular file containing PubChem CID and synoyms for chemicals (see doc on LOTUS dataset creation)", type=str, required=False, default=None, dest="cid_synonyms")
parser.add_argument("--top-n-keywords", help="Extract the top n keywords (cam be less after filtering)", type=int, required=False, default=10, dest="top_n")
parser.add_argument("--m-threads", help="Number of threads use for the model. If all the layers are put on GPU, it may be faster to set the number of threads to 1.", type=int, required=False, default=1, dest="n_threads")
parser.add_argument("--m-n-gpu", help="Number of layers of the model put on GPUs. Ps: There are 43 layers in the Vicuna-13B", type=int, required=False, default=43, dest="n_gpu")
parser.add_argument("--m-prompts-per-item", help="Number of generated prompts per items", type=int, required=False, default=10, dest="prompts_per_items")
parser.add_argument("--compressed-chem-list-labels", help="For chemical list of derivates (e.g Favolones A-D) the corresponding labels will be 'Favolones A-D' instead of the uncompressed list.", required=False, dest='compressed_chem_list_labels', action='store_true')


args = parser.parse_args()

f_input_name = os.path.splitext(os.path.basename(args.data))[0]

# INIT THE INSTRUCTION BUILDER
instruct_builder = LotusInstructionsBuilder(type="vicuna-1.1", log_file_name="instruction-builder-" + f_input_name + ".log")
logger = instruct_builder.logger

# INIT VARS
CACHE_KEYWORDS = os.path.join(args.cache_dir, "keywords.json")
syn_table = None
TH_CHEM_NAME_LEN = 60
TH_SEQ_MATCHER = 0.5
PUBTATOR = args.use_pubtator
PROMPTS_PER_ITEMS = args.prompts_per_items
N_THREADS = args.n_threads
N_GPU = args.n_gpu

# for LLAMA.CPP:
N_CTX = 2048
N_BACTH = 512
SEED = 1024
TEMPERATURE_LIST = [0.4, 0.5, 0.6, 0.7]

PROMPT = instruct_builder.get_prompt()

OUTPUT = {}

# Are chemical classes provided ?
chemical_classes = args.chemical_classes

# Cache for keywords:
if os.path.exists(CACHE_KEYWORDS):
    instruct_builder.set_keywords_cache(CACHE_KEYWORDS)
else:
    logger.info("No keywords cache file found.")

if not os.path.exists(os.path.dirname(CACHE_KEYWORDS)):
    os.makedirs(os.path.dirname(CACHE_KEYWORDS))

# Checking paths and files
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if not os.path.exists(args.data):
    logger.error("Cannot find input file at %s", args.data)
    sys.exit(1)

if args.cid_synonyms is not None:
    if not os.path.exists(args.cid_synonyms):
        logger.error("An CID-synonyms file was provided but cannot be found at %s", args.cid_synonyms)
        sys.exit(1)
    else:
        logger.info("Reading synonym table at %s", args.cid_synonyms)
        syn_table = pd.read_csv(args.cid_synonyms, header=None, names=["CID", "synonym"], sep="\t", dtype=object).dropna().reset_index()

if chemical_classes is not None:
    if not os.path.exists(args.chemical_classes):
        logger.error("A chemical-classes file was provided but cannot be found at %s", args.chemical_classes)
        sys.exit(1)
    else:
        logger.info("Reading chemical-classes file at %s", args.chemical_classes)
        chemical_classes = pd.read_csv(args.chemical_classes, sep="\t", dtype=object).dropna()

logger.info("Reading data file at %s", args.data)
with open(args.data, "r", encoding="utf-8") as f_data:
    data = json.load(f_data)

# Init http pool manager for Pubtator
retries = Retry(total=10, backoff_factor=0.1, connect=5, read=2, redirect=5, status_forcelist=[429, 500, 502, 503, 504])
http = PoolManager(retries=retries, timeout=120)

# Init the model for keywords extraction
model = None
if args.method == "KeyBERT":
    sentence_model = SentenceTransformer(args.model_path)
    model = KeyBERT(model=sentence_model)
else:
    model = Llama(model_path=args.model_path, n_ctx=N_CTX, n_batch=N_BACTH, n_threads=N_THREADS, n_gpu_layers=N_GPU, seed=SEED)

# Browse items, extract keywords, verbalise main findings and build instruction prompt
for id, item in tqdm(data.items()):
    
    pmid = item["PMID"]
    new_item = deepcopy(item)
    
    # the exclusion list:
    logger.info("PMID %s - get exclusion list", pmid)
    exclusion_list = get_exclusion_list(item=item, syn_table=syn_table, do_pubtator=PUBTATOR, seq_len_th=TH_CHEM_NAME_LEN, http_con=http)

    # Extract keywords with the keywords_extraction method
    logger.info("PMID %s - extract keywords", pmid)
    keywords = instruct_builder.keywords_extraction(item=item, model=model, to_exclude=exclusion_list, logger=logger, top_n=args.top_n, th_ratio=TH_SEQ_MATCHER, temperature_list=TEMPERATURE_LIST)
    
    # Extract main findings:
    logger.info("PMID %s - extract and verbalised main findings", pmid)
    generated_main_findings = instruct_builder.main_findings_verbaliser(item=item, n=PROMPTS_PER_ITEMS, chemical_classes=chemical_classes, compressed_labels=args.compressed_chem_list_labels)
    
    new_item.pop("relations")

    new_item["keywords"] = keywords
    prompt_list = []

    logger.info("PMID %s - serialize prompts", pmid)
    
    for mf in generated_main_findings:
        p = {}
        p["prompt"] = PROMPT.format(title=item["ArticleTitle"], keywords=', '.join(keywords) + '.', main_findings=mf.main_findings)
        p["relations"] = mf.labels
        p["organisms"] = mf.all_org_mentions
        p["chemicals"] = mf.all_chem_mentions

        prompt_list.append(p)
    
    new_item["prompts"] = prompt_list
    
    OUTPUT[pmid] = new_item

    # Save cache for keywords (for the moment at each step !):
    instruct_builder.export_keywords_cache(CACHE_KEYWORDS)

logger.info("Export results")
with open(os.path.join(args.out_dir, "prompts_" + os.path.basename(args.data)), 'w', encoding="utf-8") as out_file:
    json.dump(OUTPUT, out_file, indent=4)
logger.info("End.")