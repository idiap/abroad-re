# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import argparse
import json
import logging
import glob
import pandas as pd
from tqdm import tqdm

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", help="path to the directory containing the train, valid and test files in json", type=str, required=True, dest='input_dir')
parser.add_argument("--output-dir", help="path to the output directory to write seq2re-formated files", type=str, required=True, dest='out_dir')

args = parser.parse_args()

# VARS
F_LIST = glob.glob(os.path.join(args.input_dir, "*.json"))

# set loggers
log_path = os.path.join("seq2rel_format_converter.log")
open(log_path, 'w', encoding="utf-8").close()
handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('seq2rel-format-converter')

if not os.path.exists(args.input_dir) or not os.path.isdir(args.input_dir):
    logger.error("Please, provide a valid input path to the directory (not files)")
    sys.exit(1)

if not os.path.exists(args.out_dir):
    logger.info("Create output directory")
    os.makedirs(args.out_dir)

# If these tests passed, then we parse and convert
for f in F_LIST:
    logger.info("read %s", os.path.join(args.input_dir, f))
    f_name = os.path.splitext(os.path.basename(f))[0]

    with open(f, "r", encoding="utf-8") as path:
        data = json.load(path)
        ids = list(data.keys())

        # Create empty dataframe
        tabular_converted = pd.DataFrame(columns=["input", "labels", "null"], dtype=object, index=ids)
        tabular_converted["null"] = "null"

        logger.info("convert %s:", f)
        # fill table
        for id in tqdm(ids):

            chems = pd.DataFrame(data[id]["chemicals"])
            orgs = pd.DataFrame(data[id]["organisms"])
            # text
            input_text = data[id]["ArticleTitle"].strip("\n ") + " " + data[id]["AbstractText"].strip("\n ")
            input_text = input_text.replace("\n", "")
            input_text = input_text.replace("\t", " ")
            tabular_converted.loc[id, "input"] = input_text
            tabular_converted.loc[id, "labels"] = " ".join([orgs[orgs["id"] == org_id]["label"].values[0] + " @SPECIES@ " + chems[chems["id"] == chem_id]["label"].values[0] + " @CHEMICAL@ @SCR@" for (org_id, chem_id) in data[id]["relations"]])

    logger.info("Export to %s", os.path.join(args.out_dir, f_name + ".tsv"))
    # export
    tabular_converted.to_csv(os.path.join(args.out_dir, f_name + ".tsv"), index=False, header=False, sep="\t", columns=["input", "labels", "null"])















