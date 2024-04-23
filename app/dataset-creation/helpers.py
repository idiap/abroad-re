# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""helpers functions"""
import os
import sys
import random
import logging
import requests
import http
import pandas as pd
import numpy as np
import defusedxml.ElementTree as ET
import xmltodict

from multiprocessing import Pool
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError
from urllib3.util import Retry

from ratelimit import limits
from ratelimit import sleep_and_retry



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



def get_std_logger(name: str, path: str, level: int, stdout: bool):
    """Create a default logger

    Returns:
        logging.logger: a default logger
    """

    # set loggers
    log_path = os.path.join(path, name + ".log")
    open(log_path, "w", encoding="utf-8").close()

    logFormatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    if stdout:
        log_file_handler = logging.FileHandler(filename=log_path)
        log_file_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_file_handler)
        log_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        log_stdout_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_stdout_handler)
    else:
        log_file_handler = logging.FileHandler(filename=log_path)
        log_file_handler.setFormatter(fmt=logFormatter)
        logger.addHandler(log_file_handler)
    return logger



class PubMedFetcher:
    def __init__(self, apikey: str, email: str, name="ABRoad-PubMed-fetcher", verbose=True, **kwargs):
        self.name = name
        self.http = PoolManager(
            retries=Retry(
                **kwargs.pop(
                    "retries_args",
                    {
                        "total": 20,
                        "backoff_factor": 1,
                        "connect": 10,
                        "read": 5,
                        "redirect": 5,
                        "status_forcelist": tuple(range(401, 600)),
                    },
                )
            ),
            timeout=300,
        )
        self.logger = get_std_logger(
            name=name,
            path=kwargs.get("logging_path", "."),
            level=(logging.DEBUG if verbose else logging.INFO),
            stdout=kwargs.get("logger_stdout", True),
        )
        self.apikey = apikey
        self.email = email

    @sleep_and_retry
    @limits(calls=3, period=1)
    def send_and_parse_request(self, url, tojson=True):
        try:
            response = self.http.request("GET", url)
        except HTTPError as fail:
            self.logger.error("Request failed with error: %s", str(fail))
            return None
        except http.client.HTTPException as fail_2:
            self.logger.error("Request failed with error: %s", str(fail_2))
            return None

        if not tojson:
            return response.data.decode()

        try:
            parsed = xmltodict.parse(response.data.decode())
        except xmltodict.expat.ExpatError as fail:
            self.logger.error("Fail to parse response: %s", str(fail))
            return None

        return parsed

    def title_and_abstracts(self, list_of_pmids: list, rate:int = 100):
        BASE_URL = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
            "&id={pmids}&retmode=xml&retmax={retmax}&api_key={apikey}"
        )
        OUT = dict()

        RATE_LIMIT = rate

        workling_list = list_of_pmids.copy()

        while len(workling_list):
            if len(workling_list) > RATE_LIMIT:
                ids = workling_list[:RATE_LIMIT].copy()
                del workling_list[:RATE_LIMIT]

            else:
                ids = workling_list.copy()
                del workling_list[: len(workling_list)]

            # build and send
            url = BASE_URL.format(pmids=",".join(ids), retmax=RATE_LIMIT, apikey=self.apikey)
            output = self.send_and_parse_request(url, tojson=False)

            # if failed, replace by empty result
            if output is None:
                output = "<PubmedArticleSet></PubmedArticleSet>"

            # post-process
            list_of_tags_to_remove = ["i", "u", "b", "sup", "sub"]

            if any(["<%s>" % tag in str(output) for tag in list_of_tags_to_remove]):
                for tag in list_of_tags_to_remove:
                    output = output.replace("<%s>" % tag, "")
                    output = output.replace("</%s>" % tag, "")

            root = ET.fromstring(output)

            pubmed_article_list = root.findall("PubmedArticle")

            for pubmed_article in pubmed_article_list:
                sub = dict()
                if pubmed_article.find("MedlineCitation").find("PMID") is not None:
                    pmid = pubmed_article.find("MedlineCitation").find("PMID").text
                    sub["PMID"] = pmid
                else:
                    self.logger.warning("No PMID found for pmid %s", pmid)
                    continue

                if pubmed_article.find("MedlineCitation").find("Article").find("ArticleTitle") is not None:
                    sub["ArticleTitle"] = (
                        pubmed_article.find("MedlineCitation").find("Article").find("ArticleTitle").text
                    )
                else:
                    self.logger.warning("No article title found for pmid %s", pmid)
                    sub["ArticleTitle"] = ""

                if pubmed_article.find("MedlineCitation").find("Article").find("Abstract") is not None:
                    try:
                        sub["AbstractText"] = " ".join(
                            [
                                sub_abstract.text
                                for sub_abstract in pubmed_article.find("MedlineCitation")
                                .find("Article")
                                .find("Abstract")
                                .findall("AbstractText")
                            ]
                        )
                    except Exception as e:
                        self.logger.error(
                            "Some unexpected error happened during the parsing of %s annotations: %s",
                            sub["PMID"],
                            str(e),
                        )

                else:
                    self.logger.warning("No article abstract found for pmid %s", pmid)
                    sub["AbstractText"] = ""

                # If everything if ok, add it.
                OUT[pmid] = sub

        extracted_pmids = set(OUT.keys())

        missing_pmids = []
        if set(list_of_pmids) != extracted_pmids:
            missing_pmids = list(set(list_of_pmids) - extracted_pmids)
            self.logger.warning("Not all PMID were extracted. Missing PMIDs: %s", ", ".join(missing_pmids))

        return (OUT, missing_pmids)
