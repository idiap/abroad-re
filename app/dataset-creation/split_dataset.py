# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spliting the dataset"""
import os
import sys
import logging
import argparse
import json
import random
import numpy as np
from helpers import read_data, get_random_sample

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="path to the dataset.json file created with get_abstract_and_format.py", type=str, required=True, dest='dataset_path')
parser.add_argument("--out-dir", help="path to the output directory", type=str, required=True, dest='out_dir')
parser.add_argument("--split-mode", help="method to split the dataset", type=str, choices=["entropy", "std"], required=True, dest='split_mode', default="std")
parser.add_argument("--entropy-dir", help="path to the directory containing the log entropy files from create_lotus_dataset", type=str, required=False, dest='entropy_dir', default="")
parser.add_argument("--top-n", help="extract the top n DOI per kingdom according to entropy values", type=int, required=False, dest='topn', default=50)
parser.add_argument("--split-prop", help="split proportion for training:valid:test, e.g: 80:10:10. If split-mode='entropy', please set the test proportion to 0 and specify the miw between train and validation accordingly, e.g: 80:20:0", type=str, required=False, default="80:10:10", dest='split_prop')
parser.add_argument("--seed", help="seed", type=int, required=False, dest='seed', default=1024)

# Get arguments
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# set loggers
log_path = os.path.join(args.out_dir, "spliting_dataset.log")
open(log_path, 'w', encoding="utf-8").close()
handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('lotus-split-dataset')


# loading data
assert os.path.isfile(args.dataset_path)
train_set = None
PMIDS_FOR_TEST = []
PMIDS_FOR_VALIDATION = []

# The full dataset is initially considered to be the train set.
with open(args.dataset_path, "r", encoding="utf-8") as f_dataset:
    train_set = json.load(f_dataset)

valid_set = {}
test_set = {}
N = len(train_set)
P_TRAIN, P_VAL, P_TEST = args.split_prop.split(":")


if args.split_mode == "entropy" and P_TEST != 0:
    logger.error("If split-mode is entropy, then the proportion for the test set should be set to 0 in 'split_prop' as the number of items will be determined from 'top-n'.")

if args.split_mode == "std":
    assert float(P_TRAIN) > 0 and float(P_VAL) > 0 and float(P_TEST) >= 0

if args.split_mode == "entropy":
    if args.entropy_dir == "" or not os.path.isdir(args.entropy_dir):
        logger.error("When split-type = 'entropy', --entropy-dir need a valid value")
        sys.exit(1)

    logger.info("Use Entropy-selector to split the dataset")
    logger.info("Entropy directory: %s", args.entropy_dir)    
    logger.info("The top %d DOI will be extracted", args.topn)

    # Input vars
    dois_for_test = []
    all_dois_IN_dataset = set([t["doi"] for t in train_set.values()])

    # Create PMID (the key in the json) to DOI (the id in entropy) map:
    logger.info("Create the map DOI to PMID")
    doi_pmid_map = dict([(t["doi"], t["PMID"]) for t in train_set.values()])

    # Read the entropy files and fill the list of doi to keep for test:
    for f_entropy in os.listdir(args.entropy_dir):

        sub_d_e = read_data(path=os.path.join(args.entropy_dir, f_entropy), sep="\t", logger=logger)

        # But, some DOI in the top N may not have been integrated in the dataset because no abstract was available, so remove them before selecting the top.
        sub_d_e = sub_d_e[sub_d_e["reference_doi"].isin(all_dois_IN_dataset)]

        dois_for_test += sub_d_e[:args.topn]["reference_doi"].tolist()

    # get distinct set (because a doi could be in the top N for more than one kingdom):
    dois_for_test = set(dois_for_test)

    PMIDS_FOR_TEST = [doi_pmid_map[doi] for doi in dois_for_test]


else:
    logger.info("Use Standard-selector to split the dataset")
    logger.info("Split proportion: %s ", args.split_prop)

    n_test = int((N * float(P_TEST))/100)
    PMIDS_FOR_TEST = get_random_sample(list(train_set.keys()), n_test, args.seed)

logger.info("Number of distinct DOI selected for test (Warning: some DOI can be in the top of > 1 kingdom): %d", len(PMIDS_FOR_TEST))

# 1) Create the test set:
for pmid in PMIDS_FOR_TEST:
    test_set[pmid] = train_set.pop(pmid)

# 2) Create the validation set from the remaining data.

# If the split mode is 'std', simply use the corresponding proportion
if args.split_mode == "std":
    n_val = int((N * float(P_VAL))/100)

# If the split mode is 'entropy' then the proportion for validation is
else:
    # Use the current len of the training set.
    N = len(train_set)
    n_val = int((N * float(P_VAL))/100)

PMIDS_FOR_VALIDATION = get_random_sample(list(train_set.keys()), n_val, args.seed)
for pmid in PMIDS_FOR_VALIDATION:
    valid_set[pmid] = train_set.pop(pmid)

logger.info("TRAIN SET: %d items", len(train_set))
logger.info("VALID SET: %d items", len(valid_set))
logger.info("TEST SET: %d items", len(test_set))

logger.info("Export files at %s", args.out_dir)

with open(os.path.join(args.out_dir, 'train.json'), 'w', encoding="utf-8") as out_train_file:
    json.dump(train_set, out_train_file, indent=4)

with open(os.path.join(args.out_dir, 'valid.json'), 'w', encoding="utf-8") as out_valid_file:
    json.dump(valid_set, out_valid_file, indent=4)

with open(os.path.join(args.out_dir, 'test.json'), 'w', encoding="utf-8") as out_test_file:
    json.dump(test_set, out_test_file, indent=4)