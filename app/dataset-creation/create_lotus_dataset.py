# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Creation of the LOTUS dataset"""
import os
import sys
import logging
import argparse
from helpers import read_data, display_stat, set_seed, random_sampler, top_n_sampler, get_std_logger
from gme.gme import GreedyMaximumEntropySampler

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lotus-data", help="path to the processed lotus data file (see preprocessing.py)", type=str, required=True, dest='input')
parser.add_argument("--out-dir", help="path to the output directory", type=str, required=True, dest='out_dir')
parser.add_argument("-N", help="total number of examples per stratification. if set to -1, it will compute the sampling up to N to get entropy distribution.", type=int, required=False, dest='N', default=1000)
parser.add_argument("--sampler", help="sampling method", type=str, required=False, dest='sampler', choices=["random", "topn_rel", "topn_struct", "topn_org", "topn_sum", "GME_sum", "GME_dutopia"], default="GME_dutopia")
parser.add_argument("--seed", help="seed", type=int, required=False, dest='seed', default=1024)
parser.add_argument("--use-freq", help="Use the observed frequency (nb or relation involving an org or a chem). If not, observations are binarised. Only available for GME ", required=False, action='store_false', dest='binarised', default=True)
parser.add_argument("--n-workers", help="number of workers to work in parralel", required=False, type=int, dest='n_workers', default=1)
parser.add_argument("--approx", help="Instead of computing all the emtropy values, approximate the best candidate by only taking the best one over a random sample of n items. If 0, no approximation.", required=False, type=int, dest='approx', default=0)


args = parser.parse_args()

set_seed(args.seed)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

if not os.path.exists(os.path.join(args.out_dir, "entropies")):
    os.makedirs(os.path.join(args.out_dir, "entropies"))

# set loggers
logger = get_std_logger("lotus-dataset-creator", path=args.out_dir, level=logging.DEBUG, stdout=True)

data = read_data(path=args.input, sep="\t", logger=logger)
display_stat(data, logger)

logger.info("Iterate over stratification")

sample = []

for name, group in data.groupby("organism_taxonomy_02kingdom"):

    # Since all publication are not focused on 1 kingdom there can be relations integrated with others kingdoms in the final set without a strict filtering on DOI exclusively relation to q unique kingdom.
    print("# " + name)
    display_stat(group, logger)
    n = len(set(group["reference_doi"])) if args.N == -1 else args.N
    logger.info("sampling n = %d", n)

    if args.sampler == "random":
        sample += random_sampler(data=group, N=n, logger=logger, seed=args.seed)

    if args.sampler.startswith("topn"):
        topn_option = args.sampler.split("_")[1]
        sample += top_n_sampler(data=group, topn_type=topn_option, N=n, logger=logger, seed=args.seed)

    if args.sampler.startswith("GME"):
        gme_option = args.sampler.split("_")[1]
        gme = GreedyMaximumEntropySampler(selector=gme_option, binarised=args.binarised, logger=logger, n_workers=args.n_workers)
        out_entropy = gme.sample(data=group, N=args.N, item_column="reference_doi", on_columns=["organism_wikidata", "structure_wikidata"], approx=args.approx)
        out_entropy.to_csv(os.path.join(args.out_dir, "entropies", "e_" + name + "_" + args.sampler +  ("_freq" if not args.binarised else "") + ".tsv"), sep="\t", index=False)
        sample += out_entropy["reference_doi"].tolist()

# Since one "generalist" article (but unexpected since we limited the umber of relations) can maximised H in different subset of kindoms, the final number of doi may be lower than N * nb_kingdom
out_data = data[data["reference_doi"].isin(sample)]
out_data.to_csv(os.path.join(args.out_dir, "sample_" + args.sampler +  ("_freq" if not args.binarised else "") + ".tsv"), sep="\t", index=False)
