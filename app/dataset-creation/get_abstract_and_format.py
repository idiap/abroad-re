# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import argparse
import json
import json
import numpy as np
from helpers import read_data, PubMedFetcher
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to data file to format", type=str, required=True, dest='input')
parser.add_argument("--ncbi-apikey", help="NCBI api Key", type=str, required=True, default="", dest='apikey')
parser.add_argument("--e-mail", help="The e-mail", type=str, required=True, default="", dest='email')
parser.add_argument("--chunk-size", help="chunk isze for requesting pubmed api", type=int, required=False, default=100, dest='chunk_size')
parser.add_argument("--out-dir", help="path to the output directory", type=str, required=True, dest='out_dir')

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

with open(os.path.join(args.out_dir, 'request_fails.txt'), 'w', encoding="utf-8"):
    pass

# set loggers
pubmed_fetcher = PubMedFetcher(
    apikey=args.apikey,   # "3887d5b3d82fd58159782a805cc40251f008"
    email=args.email,     # "mdelmas@idiap.ch"
    verbose=True,
    logging_path=args.out_dir,
    logger_stdout=False,
)

data = read_data(args.input, sep="\t", logger=pubmed_fetcher.logger)

all_dois = list(set(data["reference_doi"]))
all_pmids = list(set(data["reference_pubmed_id"]))

assert len(all_dois) == len(all_pmids)

reference_table = data[["reference_wikidata", "reference_doi", "reference_pubmed_id"]].drop_duplicates()

final_dict, all_missing_pmids = pubmed_fetcher.title_and_abstracts(list_of_pmids=all_pmids, rate=args.chunk_size)

pubmed_fetcher.logger.info("Fill with relations info")
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

pubmed_fetcher.logger.info("All missing PMID: %s", ",".join(all_missing_pmids))

with open(os.path.join(args.out_dir, 'dataset.json'), 'w', encoding="utf-8") as outfile:
    json.dump(final_dict, outfile, indent=4)
