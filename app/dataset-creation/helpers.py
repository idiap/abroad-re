# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""helpers functions"""
import sys
import random
import logging
import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

from multiprocessing import Pool

def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

def read_data(path:str , sep:str, logger:logging.Logger) -> pd.DataFrame:
    """Helper to read a dataframe

    Args:
        path (str): path to the file
        sep (str): separator
        logger (logging.Logger): logger instance

    Returns:
        pd.DataFrame: the dataframe
    """

    logger.info("Read file at %s", path)

    try:
        data = pd.read_csv(path, sep=sep, dtype=object).fillna("")

    except pd.errors.ParserError as except_parsing_error:
        logger.error("file at %s has incorrect format. %s", path, str(except_parsing_error))
        sys.exit(1)

    except FileNotFoundError:
        logger.error("File not found at %s", path)
        sys.exit(1)

    return data

def display_stat(data: pd.DataFrame, logger:logging.Logger):
    """Display statistics: number of distinct chenicals, organisms and references in a dataset.

    Args:
        data (pd.DataFrame): the dataset
    """
    d = {"n_items": data.shape[0], 
        "n_orgamisms": len(set(data["organism_wikidata"])),
        "n_chemicals": len(set(data["structure_wikidata"])),
        "n_references": len(set(data["reference_doi"]))
        }
    logger.info(f"- Number of items: {d['n_items']}")
    logger.info(f"- Number of organisms: {d['n_orgamisms']}")
    logger.info(f"- Number of chemicals: {d['n_chemicals']}")
    logger.info(f"- Number of references: {d['n_references']}")

def random_sampler(data:pd.DataFrame, N:int, logger:logging.Logger, seed:int) -> list:
    """Randomly select N references (DOI) from a dataset

    Args:
        data (pd.DataFrame): the dataset
        N (int): the number of references (DOI) to sample
        logger (logging.Logger): the logger instance
        seed (int): the random seed

    Returns:
        list: a list containing the sampled DOI
    """
    dois = list(set(data["reference_doi"]))

    if N > len(dois):
        logger.warning("N=%d greater than the total number of available DOI (%d). Returning the total set.", N, len(dois))
        return dois

    random.seed(seed)
    sample = random.sample(sorted(dois), N)

    return sample

def top_n_sampler(data:pd.DataFrame, topn_type:str, N:int, logger:logging.Logger, seed:int) -> pd.DataFrame:
    """Select the top N references (DOI) providing the most relations.

    Args:
        data (pd.DataFrame): the dataset
        topn_type (str): which type of top n ? on relations (rel), structure (struct) or organisms (org) ?
        N (int): the number of references (DOI) to extract
        logger (logging.Logger): the logger instance
        seed (int): the random seed

    Returns:
        pd.DataFrame: a list containing the selected DOI
    """
    topn_data = data.copy()

    if topn_type == "rel":
        topn_data["rel_id"] = topn_data["structure_wikidata"] + "-" + topn_data["organism_wikidata"]
        topn_data = topn_data.groupby("reference_doi")["rel_id"].agg(N="nunique").reset_index()

    elif topn_type == "struct":
        topn_data = topn_data.groupby("reference_doi")["structure_wikidata"].agg(N="nunique").reset_index()

    elif topn_type == "org":
            topn_data = topn_data.groupby("reference_doi")["organism_wikidata"].agg(N="nunique").reset_index()

    elif topn_type == "sum": 
        topn_data_1 = topn_data.groupby("reference_doi")["structure_wikidata"].agg(N="nunique").reset_index()
        topn_data_2 = topn_data.groupby("reference_doi")["organism_wikidata"].agg(N="nunique").reset_index()

        assert all(topn_data_1.reference_doi == topn_data_2.reference_doi)

        topn_data = pd.DataFrame({"reference_doi": topn_data_1["reference_doi"], "N": (topn_data_1["N"] + topn_data_2["N"])})

    else:
        logger.error("Top n sampler option %s does not exist.", topn_type)
        sys.exit(1)

    sample = []

    dois = topn_data["reference_doi"].tolist()
    counts = topn_data["N"].tolist()

    if N > len(dois):
        logger.warning("N=%d greater than the total number of available DOI (%d). Returning the total set.", N, len(dois))
        return dois

    np.random.seed(seed)
    ordered_indexes = np.array(counts).argsort()[::-1][:N]

    for index in ordered_indexes:
        sample.append(dois[index])

    return sample

def get_pubmed_data(http, ids:list, api_key:str, logger:logging.Logger) -> tuple[dict, list]:
    """Extract PubMed data (Title + abstract) for a list of PubMed ids

    Args:
        http (urllib3.poolmanager.PoolManage): The http pool Manager for sending the request to the PubMed EUtils API.
        ids (list): the lsit of PubMed ids.
        api_key (str): The NCBI account API key.
        logger (logging.Logger): the logger instance

    Returns:
        dict, list: out, missing_pmids. out is a dictionnary of pubmed data. Keys are PubMed ids and values the corresponding titles and abstract. missing_pmids is the list of PubMed ids for which these information could not have been found.
    """
    try:

        response = http.request("GET",'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=%s&retmode=xml&retmax=100000&usehistory=n&api_key=%s' % (','.join(ids), api_key), headers = {"Content-Type": "application/xml"})

    except requests.exceptions.RequestException as fail_request:
        logger.error("Request failed with error: %s", str(fail_request))
        return (False, False)

    # test if html tags are present and if so remove them
    list_of_tags_to_remove = ["i", "u", "b", "sup", "sub"]

    response = response.data.decode()

    if any(["<%s>" % tag in str(response) for tag in list_of_tags_to_remove]):
        for tag in list_of_tags_to_remove:
            response = response.replace("<%s>" % tag, "")
            response = response.replace("</%s>" % tag, "")

    root = ET.fromstring(response)

    out = dict()

    pubmed_article_list = root.findall("PubmedArticle")

    for i, pubmed_article in enumerate(pubmed_article_list):

        sub = dict()
        if pubmed_article.find("MedlineCitation").find("PMID") is not None:
            sub["PMID"] = pubmed_article.find("MedlineCitation").find("PMID").text
        else:
            logger.warning("No PMID found for item %d", i)
            continue

        if pubmed_article.find("MedlineCitation").find("Article").find("ArticleTitle") is not None:
            sub["ArticleTitle"] = pubmed_article.find("MedlineCitation").find("Article").find("ArticleTitle").text
        else:
            logger.warning("No article title found for item %d", i)
            continue

        if pubmed_article.find("MedlineCitation").find("Article").find("Abstract") is not None:
            sub["AbstractText"] = " ".join([sub_abstract.text for sub_abstract in pubmed_article.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")])
        else:
            logger.warning("No article abstract found for item %d", i)
            continue

        # If everything if ok, add it.
        out[sub["PMID"]] = sub

    extracted_pmids = set(out.keys())

    missing_pmids = []
    if set(ids) != extracted_pmids:
        missing_pmids = list(set(ids) - extracted_pmids)
        logger.warning("Not all PMID were extracted. Missing PMID: %s", ", ".join(missing_pmids))

    return (out, missing_pmids)

def get_random_sample(list_of_pmids:list, n:int, seed:int) -> list:
    """Create a random saple of PMIDs

    Args:
        list_of_pmids (list): a lsit of PMID
        n (int): size of the sample
        seed (int): random seed

    Returns:
        list: the list of sampled PMID
    """
    random.seed(seed)
    sample = random.sample(list_of_pmids, n)
    return sample