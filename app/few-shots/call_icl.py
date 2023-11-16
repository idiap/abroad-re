# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Creation of the LOTUS dataset"""
import os
import sys
import argparse
import logging
from llama_cpp import Llama
import json

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="path to the model file", type=str, required=True, dest='model_path')
parser.add_argument("--input-file", help="path to the data test file", type=str, required=True, dest='data')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='out_dir')
parser.add_argument("--prompt-type", help="Classic or instructions-tuned prompts", type=str, required=True, dest='prompt_type', choices=["classic", "instruct"])
parser.add_argument("--seed", help="seed", type=int, required=False, dest='seed', default=1024)
parser.add_argument("--m-threads", help="Number of threads use for the model. If all the layers are put on GPU, it may be faster to set the number of threads to 1.", type=int, required=False, default=1, dest="n_threads")
parser.add_argument("--m-n-gpu", help="Number of layers of the model put on GPUs. Ps: There are 43 layers in the Vicuna-13B", type=int, required=False, default=43, dest="n_gpu")
parser.add_argument("--temperature", help="temperature used whn generating the answer.", required=False, dest='temperature', default=0)

args = parser.parse_args()


def get_logger(name, **kwargs):
    """Create a default logger

    Returns:
        logging.logger: a default logger
    """

    # set loggers
    level = kwargs.get("level", logging.INFO)
    log_path = kwargs.get("path", "./prompting-generate-abstract.log")
    open(log_path, 'w', encoding="utf-8").close()

    handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=level)
    logger = logging.getLogger(name)

    return logger

# set batch and n_ctx to max:
N_CTX = 2048
N_BACTH = 1024
N_THREADS = args.n_threads
N_GPU = args.n_gpu

# For generating
STOPS = ["INPUT", "OUTPUT"]
MAX_TOKENS = 300
OUTPUT = {}

# INIT THE ICL PROMPT (until ~ 1000 tokens)
# FROM PMID: 11557080, 25915800, 16843570, 9868162, 21104530
PROMPT_CLASSIC = """The task is to extract relations between organisms and chemicals from the input text.
INPUT: The antimicrobially active EtOH extracts of Maytenus heterophylla yielded a new dihydroagarofuran alkaloid, 1beta-acetoxy-9alpha-benzoyloxy-2beta,6alpha-dinicotinoyloxy-beta-dihydroagarofuran, together with the known compounds beta-amyrin, maytenfolic acid, 3alpha-hydroxy-2-oxofriedelane-20alpha-carboxylic acid, lup-20(29)-ene-1beta,3beta-diol, (-)-4'-methylepigallocatechin, and (-)-epicatechin.
OUTPUT: Maytenus heterophylla produces 1beta-acetoxy-9alpha-benzoyloxy-2beta,6alpha-dinicotinoyloxy-beta-dihydroagarofuran. Maytenus heterophylla produces beta-amyrin. Maytenus heterophylla produces maytenfolic acid. Maytenus heterophylla produces 3alpha-hydroxy-2-oxofriedelane-20alpha-carboxylic acid. Maytenus heterophylla produces lup-20(29)-ene-1beta,3beta-diol. Maytenus heterophylla produces (-)-4'-methylepigallocatechin. Maytenus heterophylla produces (-)-epicatechin.
INPUT: Ten new ergosteroids, gloeophyllins A-J (1-10), have been isolated from the solid cultures of Gloeophyllum abietinum.
OUTPUT: Gloeophyllum abietinum produces gloeophyllin A. Gloeophyllum abietinum produces gloeophyllin B. Gloeophyllum abietinum produces gloeophyllin C. Gloeophyllum abietinum produces gloeophyllin D. Gloeophyllum abietinum produces gloeophyllin E. Gloeophyllum abietinum produces gloeophyllin F. Gloeophyllum abietinum produces gloeophyllin G. Gloeophyllum abietinum produces gloeophyllin H. Gloeophyllum abietinum produces gloeophyllin I. Gloeophyllum abietinum produces gloeophyllin J.
INPUT: The present work describes the isolation of the cyclic peptides geodiamolides A, B, H and I (1-4) from G. corticostylifera and their anti-proliferative effects against sea urchin eggs and human breast cancer cell lineages.
OUTPUT: G. corticostylifera produces geodiamolide A. G. corticostylifera produces geodiamolide B. G. corticostylifera produces geodiamolide H. G. corticostylifera produces geodiamolide I.
INPUT: Four new cyclic peptides, patellamide G (2) and ulithiacyclamides E-G (3-5), along with the known patellamides A-C (6-8) and ulithiacyclamide B (9), were isolated from the ascidian Lissoclinum patella collected in Pohnpei, Federated States of Micronesia.
OUTPUT: Lissoclinum patella produces patellamide G. Lissoclinum patella produces ulithiacyclamide E. Lissoclinum patella produces ulithiacyclamide F. Lissoclinum patella produces ulithiacyclamide G. Lissoclinum patella produces patellamide A. Lissoclinum patella produce patellamide B. Lissoclinum patella produces patellamide C. Lissoclinum patella produces ulithiacyclamide B.
INPUT: Chemical investigation of Trogopterus faeces has led to the isolation of seven flavonoids. Their structures were elucidated by chemical and spectral analyses. In an anticoagulative assay, three kaempferol coumaroyl rhamnosides had significant antithrombin activity. This is the first report on the occurrence of flavonoid glycosides in Trogopterus faeces.
OUTPUT: Trogopterus faeces produces flavonoids. Trogopterus faeces produces kaempferol coumaroyl rhamnosides. Trogopterus faeces produces flavonoid glycosides.
INPUT: %s
OUTPUT: """

PROMPT_INSTRUCT = """Instructions: The task is to extract relations between organisms and chemicals from the input text.
INPUT: The antimicrobially active EtOH extracts of Maytenus heterophylla yielded a new dihydroagarofuran alkaloid, 1beta-acetoxy-9alpha-benzoyloxy-2beta,6alpha-dinicotinoyloxy-beta-dihydroagarofuran, together with the known compounds beta-amyrin, maytenfolic acid, 3alpha-hydroxy-2-oxofriedelane-20alpha-carboxylic acid, lup-20(29)-ene-1beta,3beta-diol, (-)-4'-methylepigallocatechin, and (-)-epicatechin.
OUTPUT: Maytenus heterophylla produces 1beta-acetoxy-9alpha-benzoyloxy-2beta,6alpha-dinicotinoyloxy-beta-dihydroagarofuran. Maytenus heterophylla produces beta-amyrin. Maytenus heterophylla produces maytenfolic acid. Maytenus heterophylla produces 3alpha-hydroxy-2-oxofriedelane-20alpha-carboxylic acid. Maytenus heterophylla produces lup-20(29)-ene-1beta,3beta-diol. Maytenus heterophylla produces (-)-4'-methylepigallocatechin. Maytenus heterophylla produces (-)-epicatechin.
INPUT: Ten new ergosteroids, gloeophyllins A-J (1-10), have been isolated from the solid cultures of Gloeophyllum abietinum.
OUTPUT: Gloeophyllum abietinum produces gloeophyllin A. Gloeophyllum abietinum produces gloeophyllin B. Gloeophyllum abietinum produces gloeophyllin C. Gloeophyllum abietinum produces gloeophyllin D. Gloeophyllum abietinum produces gloeophyllin E. Gloeophyllum abietinum produces gloeophyllin F. Gloeophyllum abietinum produces gloeophyllin G. Gloeophyllum abietinum produces gloeophyllin H. Gloeophyllum abietinum produces gloeophyllin I. Gloeophyllum abietinum produces gloeophyllin J.
INPUT: The present work describes the isolation of the cyclic peptides geodiamolides A, B, H and I (1-4) from G. corticostylifera and their anti-proliferative effects against sea urchin eggs and human breast cancer cell lineages.
OUTPUT: G. corticostylifera produces geodiamolide A. G. corticostylifera produces geodiamolide B. G. corticostylifera produces geodiamolide H. G. corticostylifera produces geodiamolide I.
INPUT: Four new cyclic peptides, patellamide G (2) and ulithiacyclamides E-G (3-5), along with the known patellamides A-C (6-8) and ulithiacyclamide B (9), were isolated from the ascidian Lissoclinum patella collected in Pohnpei, Federated States of Micronesia.
OUTPUT: Lissoclinum patella produces patellamide G. Lissoclinum patella produces ulithiacyclamide E. Lissoclinum patella produces ulithiacyclamide F. Lissoclinum patella produces ulithiacyclamide G. Lissoclinum patella produces patellamide A. Lissoclinum patella produce patellamide B. Lissoclinum patella produces patellamide C. Lissoclinum patella produces ulithiacyclamide B.
INPUT: Chemical investigation of Trogopterus faeces has led to the isolation of seven flavonoids. Their structures were elucidated by chemical and spectral analyses. In an anticoagulative assay, three kaempferol coumaroyl rhamnosides had significant antithrombin activity. This is the first report on the occurrence of flavonoid glycosides in Trogopterus faeces.
OUTPUT: Trogopterus faeces produces flavonoids. Trogopterus faeces produces kaempferol coumaroyl rhamnosides. Trogopterus faeces produces flavonoid glycosides.
INPUT: %s
OUTPUT: """

# load model
if not os.path.exists(args.model_path):
    print("Can't find model file")
    sys.exit(1)

if not os.path.exists(args.data):
    print("Can't find data file")
    sys.exit(1)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open(args.data) as test_file:
    test_data = json.load(test_file)

filename = os.path.splitext(os.path.basename(args.data))[0]

logger = get_logger(name="ICL", path=os.path.join(args.out_dir, "log-icl.log"))

try:
    llm = Llama(model_path=args.model_path, n_ctx=N_CTX, n_batch=N_BACTH, n_threads=N_THREADS, n_gpu_layers=N_GPU, seed=args.seed)
except ValueError as e_init:
    logger.error(str(e_init))
    sys.exit(0)

logger.info("Temp: %.2f", args.temperature)

if args.prompt_type == "classic":
    PROMPT = PROMPT_CLASSIC
else:
    PROMPT = PROMPT_INSTRUCT

for PMID, item in test_data.items():

    out = dict()

    # Fill it into the prompt
    prompt = PROMPT %(item["ArticleTitle"].strip("\n ") + " " + item["AbstractText"].strip("\n "))

    try:
        llm.reset()
        logger.info("send prompt for PMID %s", PMID)
        out = llm(prompt, max_tokens=MAX_TOKENS, stop=STOPS, temperature=args.temperature)
        out["status"] = "ok"
        
    except ValueError as e1:
        out = {"status": "ERROR: " + str(e1)}
        logger.error("Unexpected error while promting for icl: %s", PMID)

    except RuntimeError as e2:
        out = {"status": "ERROR: " + str(e2)}
        logger.error("Unexpected error while promting for icl: %s", PMID)

    out["prompt"] = prompt
    out["PMID"] = PMID
    out["ArticleTitle"] = item["ArticleTitle"]
    out["AbstractText"] = item["AbstractText"]

    OUTPUT[PMID] = out

with open(os.path.join(args.out_dir, "output_" + filename + ".json"), "w", encoding="utf-8") as f_out:
    json.dump(OUTPUT, f_out, indent=4)
