# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import logging
import argparse
import json
import eutils
import json
import numpy as np

from urllib3.util import Retry
from urllib3 import PoolManager

from helpers import read_data, get_pubmed_data
from time import sleep
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to data file to format", type=str, required=True, dest='input')
parser.add_argument("--ncbi-apikey", help="NCBI api Key", type=str, required=False, default="", dest='apikey')
parser.add_argument("--chunk-size", help="chunk isze for requesting pubmed api", type=int, required=False, default=100, dest='chunk_size')
parser.add_argument("--out-dir", help="path to the output directory", type=str, required=True, dest='out_dir')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open(os.path.join(args.out_dir, 'request_fails.txt'), 'w', encoding="utf-8"):
    pass

# set loggers
log_path = os.path.join(args.out_dir, "formating.log")
open(log_path, 'w', encoding="utf-8").close()
handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)
logger = logging.getLogger('lotus-dataset-formating')


data = read_data(args.input, sep="\t", logger=logger)

all_dois = list(set(data["reference_doi"]))
all_pmids = list(set(data["reference_pubmed_id"]))

assert len(all_dois) == len(all_pmids)

reference_table = data[["reference_wikidata", "reference_doi", "reference_pubmed_id"]].drop_duplicates()

retries = Retry(total=10, backoff_factor=0.1, connect=5, read=2, redirect=5, status_forcelist=[429, 500, 502, 503, 504])
http = PoolManager(retries=retries, timeout=120)

n = args.chunk_size
pmids_chunks = [all_pmids[i * n:(i + 1) * n] for i in range((len(all_pmids) + n - 1) // n )]

final_dict = dict()
all_missing_pmids = []

logger.info("%d chunks to proceed", len(pmids_chunks))
for i, chunk in enumerate(pmids_chunks):
    sleep(1)
    logger.info("processing chunk %d", i)
    pmids_data, missing_pmids = get_pubmed_data(http=http, ids=chunk, api_key=args.apikey, logger=logger)

    # If the request failed despite several tries.
    if not pmids_data and not missing_pmids:
        logger.warning("Previous request as failed for unknown reasons. PubMed ids will be exported in the request_failed file.")
        with open(os.path.join(args.out_dir, 'request_fails.txt'), 'w', encoding="utf-8") as f_r_failed:
            for id in chunk:
                f_r_failed.write(id)

    # If everything ok, add
    final_dict = final_dict | pmids_data
    all_missing_pmids += missing_pmids


logger.info("Fill with relations info")
for pmid in tqdm(final_dict.keys()):

    final_dict[pmid]["id"] = reference_table[reference_table["reference_pubmed_id"] == pmid]["reference_wikidata"].values[0].split("http://www.wikidata.org/entity/")[1]
    final_dict[pmid]["doi"] = reference_table[reference_table["reference_pubmed_id"] == pmid]["reference_doi"].values[0]

    sub_data = data[data["reference_pubmed_id"] == pmid]

    # add wikidata chemical entities info
    final_dict[pmid]["chemicals"] = list()
    for i, r in sub_data[["structure_wikidata", "structure_cid", "structure_nameTraditional"]].drop_duplicates().iterrows():
        final_dict[pmid]["chemicals"].append({"id": r["structure_wikidata"].split("http://www.wikidata.org/entity/")[1], "pubchem_id": r["structure_cid"], "label": r["structure_nameTraditional"]})

    final_dict[pmid]["organisms"] = list()
    for i, r in sub_data[["organism_wikidata", "organism_name"]].drop_duplicates().iterrows():
        final_dict[pmid]["organisms"].append({"id": r["organism_wikidata"].split("http://www.wikidata.org/entity/")[1], "label": r["organism_name"]})

    final_dict[pmid]["relations"] = list()
    for i, r in sub_data[["organism_wikidata", "structure_wikidata"]].drop_duplicates().iterrows():
        final_dict[pmid]["relations"].append([r["organism_wikidata"].split("http://www.wikidata.org/entity/")[1], r["structure_wikidata"].split("http://www.wikidata.org/entity/")[1]])

logger.info("All missing PMID: %s", ",".join(all_missing_pmids))

with open(os.path.join(args.out_dir, 'dataset.json'), 'w', encoding="utf-8") as outfile:
    json.dump(final_dict, outfile, indent=4)
