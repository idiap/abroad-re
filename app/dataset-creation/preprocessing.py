# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pre-processing of the LOTUS dataset"""
import os
import sys
import logging
import argparse
from helpers import read_data, display_stat

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lotus-data", help="path to lotus data file", type=str, required=True, dest='input')
parser.add_argument("--doi2pmid", help="path to the file containing mapping betwwen DOI and PMID", type=str, required=True, dest='doi2pmid')
parser.add_argument("--out-dir", help="path to the output directory", type=str, required=True, dest='out_dir')
parser.add_argument("--max-rel-per-ref", help="maximum number of relation per document", type=int, required=False, dest='max_rel_per_doc', default=20)
parser.add_argument("--chemical-name-max-len", help="The maximal length for a chemical name. All relations involving a chemical with a label longer than this parameter will be excluded. Note that this filter is applied AFTER the one on the number of relations per DOI", type=int, required=False, dest='max_chem_name_len', default=60)

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# set loggers
log_path = os.path.join(args.out_dir, "preprocessing.log")
open(log_path, 'w', encoding="utf-8").close()
handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG)
logger = logging.getLogger('lotus-dataset-preprocessing')


data = read_data(path=args.input, sep=",", logger=logger)
data = data[["structure_wikidata", "structure_cid", "structure_nameTraditional", "organism_wikidata", "organism_name", "organism_taxonomy_02kingdom", "reference_wikidata", "reference_doi"]]
data = data.drop_duplicates()
display_stat(data, logger)

logger.info("Replace NA values in 'organism_taxonomy_02kingdom' with 'Not Attributed (Bacteria or Algae)'")
data.loc[data.organism_taxonomy_02kingdom == "", "organism_taxonomy_02kingdom"] = "Not Attributed (Bacteria or Algae)"

doi2pmid = read_data(path=args.doi2pmid, sep="\t", logger=logger)
doi2pmid = doi2pmid.drop_duplicates()

# Drop items whem there are more than 2 PMID for 1 DOI
doi2pmid.drop(doi2pmid[doi2pmid.reference_doi.duplicated()].index, inplace=True)

data = data[data['reference_doi'].isin(doi2pmid["reference_doi"].tolist())]
logger.info("Removing items without PMID")
display_stat(data, logger)

logger.info("Merge PMID and DOI identifiers")
data = data.merge(doi2pmid, on="reference_doi", how="left")
del doi2pmid

logger.info("Filtering by max number of supported relations: <= %d", args.max_rel_per_doc)
# No need to worry, there is no NA
number_of_ref_per_doi = data.copy()
number_of_ref_per_doi["rel_id"] = number_of_ref_per_doi["structure_wikidata"] + "-" + number_of_ref_per_doi["organism_wikidata"]
number_of_ref_per_doi = number_of_ref_per_doi.groupby("reference_doi")["rel_id"].agg(N="nunique").reset_index()
number_of_ref_per_doi.drop(number_of_ref_per_doi[number_of_ref_per_doi["N"] > args.max_rel_per_doc].index, inplace=True)
data = data[data['reference_doi'].isin(number_of_ref_per_doi["reference_doi"].tolist())]
display_stat(data, logger)


logger.info("Filtering by removing relations with a chemical which have a label longer than: <= %d", args.max_chem_name_len)

# We also remove at the same time those without a name ;)
data = data[(data["structure_nameTraditional"].str.len() > 0) & (data["structure_nameTraditional"].str.len() <= args.max_chem_name_len)]
display_stat(data, logger)

data.to_csv(os.path.join(args.out_dir, "processed_lotus_data.tsv"), sep="\t", index=False)
