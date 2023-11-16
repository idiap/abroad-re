# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import logging
import pandas as pd
import json
import re
import urllib3

def get_pubTator_annotations(pmid:str, http, **kwargs):

    out = []

    logger = kwargs.pop("logger", logging.getLogger())

    try:
        response = http.request("GET", 'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids=%s' % (pmid), headers = {"Content-Type": "application/json"})

    except (urllib3.exceptions.HTTPError) as fail_request:
        logger.error("Request failed with error: %s", str(fail_request))
        return []
    
    try:
        toparse = json.loads(response.data.decode())

    except json.decoder.JSONDecodeError as json_error:
        logger.warning("Error during decoding returned json from pubtator. Response looks to be empty. %s", json_error)
        return []

    for passage in toparse["passages"]:

        if passage["infons"]["type"] != "abstract":
            continue

        for annot in passage["annotations"]:
            if annot["infons"]["type"] == "Species" or annot["infons"]["type"] == "Chemical":
                out.append(annot["text"])
    
    return out


def get_exclusion_list(item:dict, syn_table:pd.DataFrame, do_pubtator:bool = True, seq_len_th:float = 0.5, **kwargs):
    
    logger = kwargs.pop("logger", logging.getLogger())

    pmid = item["PMID"]

    # Exclusion list
    toexclude = []

    # Add all organism names 
    l_org = [org["label"] for org in item["organisms"]]
    logger.debug("Add organism label(s) to exclusion list: %s ", ', '.join(l_org))
    toexclude += l_org

    # Add all chemical names
    l_chem = [chem["label"] for chem in item["chemicals"]]
    logger.debug("Add chemical label(s) to exclusion list: %s", ', '.join(l_chem))
    toexclude += l_chem    

    # If provided, add all corresponding synonyms
    if syn_table is not None:
        for cid in [chem["pubchem_id"] for chem in item["chemicals"] if "pubchem_id" in chem]:
            for syn in syn_table[syn_table["CID"] == cid]["synonym"].tolist():
                if len(syn) <= seq_len_th and not re.match("^[\sA-Z0-9-:<>=\\]\\[,\\(\\)]*$", syn):
                    logger.debug("Add synonym of cid %s to exclusion list: %s", cid, syn)
                    toexclude.append(syn)
            
    if do_pubtator:

        if "http_con" not in kwargs:
            logger.error("Please provide an http_con argument for calling requesting PubTator")
            sys.exit(1)

        http = kwargs.get("http_con")
        
        pubtator_list = get_pubTator_annotations(pmid, http, logger=logger)
        if len(pubtator_list):
            logger.debug("Add organism and/or chemical names extracted with PubTator (id %s): %s", id, ', '.join(pubtator_list))
            toexclude += pubtator_list

    
    toexclude = list(set([w.lower() for w in toexclude]))
    
    return toexclude