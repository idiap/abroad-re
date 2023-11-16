# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import logging
import keybert
import json
import re
import llama_cpp
import numpy as np
import pandas as pd

from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
from collections import defaultdict
from difflib import SequenceMatcher
from string import punctuation
from instructions_helpers import create_list_of_np, shuffle, stringify, get_relations_labels
from general_helpers import get_logger

class InstructionsBuilder(ABC):

    @abstractmethod
    def keywords_extraction(self, item:dict, *args, **kwars) -> list:
        pass

    @abstractmethod
    def main_findings_verbaliser(self, item:dict, *args, **kwars) -> list:
        pass

class LotusInstructionsBuilder(InstructionsBuilder):

    def __init__(self, type="std", logger_level=logging.DEBUG, log_dir=".", log_file_name="instruction-builder.log"):
        self.cache_keywords = {}
        self.type = type
        self.logger = get_logger(name="instruction-builder", level=logger_level, path=os.path.join(log_dir, log_file_name))
    
    def get_prompt(self):
        if self.type == "std":
            PROMPT="""
            Instructions: Given a title, a list of keywords and main findings, create an abstract for a scientific article.
            Title: {title}
            Keywords: {keywords}
            Main findings: {main_findings}
            Abstract:"""
            return PROMPT
        elif self.type == "vicuna-1.1":
            PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                      "The assistant helps write abstracts of scientific articles in biology. "
                      "The assistant must not start the created abstract with common formulations such as 'In this study', 'This article reports', 'The present study', or something similar. "
                      "USER: Given a title, a list of keywords and main findings, create an abstract for a scientific article. "
                      "Title: {title} "
                      "Keywords: {keywords} "
                      "Main findings: {main_findings} "
                      "ASSISTANT: \nAbstract:")
            return PROMPT
        else:
            self.logger.error("Unknown prompt type %s", self.type)
            sys.exit(1)


    def set_keywords_cache(self, path_to_cache_file):

        if os.path.exists(path_to_cache_file):
            self.logger.info("Open cache file at %s", path_to_cache_file)
            with open(path_to_cache_file, "r", encoding="utf-8") as f_cache:
                self.cache_keywords = json.load(f_cache)
        else:
            self.logger.info("No cache file found.")
    
    def export_keywords_cache(self, path_to_cache_file):
        
        self.logger.info("Export cache keywords")
        with open(path_to_cache_file, "w", encoding="utf-8") as f_cache_write:
            json.dump(self.cache_keywords, f_cache_write, indent=4)

    # overrides keywords_extraction:
    def keywords_extraction(self, item:dict, model, to_exclude:list, **kwargs):
        """
        Example of params argument for KeyBERT:
        params = {"keyphrase_ngram_range": (1,2), "stop_words": None, "use_mmr": True,"diversity": 0.7, "top_n": 10}

        Example of params argument for llama-like model:
        params = {prompt: 'Intructions ...', "max_tokens": 500, "repeat_penalty": 1.1, "temperature_range": [0.5, 0.5], "top_n": 10}
        """

        DEFAULT_PROMPT = """Extract a comma-separated list of keywords from the following abstract of a scientific article.
Input: %s
Keywords:"""
        
        # A list of banned token that should be removed if they are, or are contained in one keyword.
        BANNED_TOKENS = ["keyword", "\n", "\t", "abstract", "scientific article"] 
        
        # Number of keywords 
        TOP_N = 10

        # For str comparison
        DEFAULT_TH_RATIO = 0.5
        PUNCTUATION = ['!', '"', '#', '$', '&', '\', ''', '*', '+', ',', '.', '/', ':', ';', '.', '?', '^', '_', '`', '{', '|', '}', '~'] # '(', ')', '[', ']'

        # CHECKING AND INIT
        params = None
        title = item["ArticleTitle"]
        abstract = item["AbstractText"]
        doc = abstract + title
        pmid = item["PMID"]

        # The keywords dict
        d = defaultdict(lambda: 0)

        if isinstance(model, keybert._model.KeyBERT):
            self.logger.info("Using KeyBERT")
            params = kwargs.pop("params", {"keyphrase_ngram_range": (1,2), "stop_words": None, "use_mmr": True, "diversity": 0.7})

        elif isinstance(model, llama_cpp.llama.Llama):
            self.logger.info("Using llama.cpp")
            params = kwargs.pop("params", {"max_tokens": 500, "repeat_penalty": 1.1})
        else:
            self.logger.error("Not supported model type")
            sys.exit(1)
        
        # Init from args or defaults
        th_ratio = kwargs.pop("th_ratio", DEFAULT_TH_RATIO)
        top_n = kwargs.pop("top_n", TOP_N)

        # RUN FOR KEYBERT
        if isinstance(model, keybert._model.KeyBERT):
            params["top_n"] = top_n
            out = model.extract_keywords(doc, **params)
            d = dict(out)
        
        # RUN FOR LLAMA.CPP
        else:

            # Check in cache:
            if pmid in self.cache_keywords:
                self.logger.info("PMID %s: retrieve list of keywords from cache.", pmid)
                d = self.cache_keywords[pmid]

            # If not in cache
            else:

                self.logger.info("PMID %s not found in cache.", pmid)

                prompt = kwargs.pop("prompt", DEFAULT_PROMPT)
                prompt = prompt %(doc)

                temperature_list = kwargs.pop("temperature_list", [0.4, 0.5])

                # On all temperatures 
                for temperature in temperature_list:

                    # generate a random temperature from the  temperature_range params for the sample
                    params["temperature"] = temperature
                    self.logger.info("temp: %.2f", params["temperature"])

                    # Send the prompt
                    self.logger.debug("Prompt: %s", prompt)
                    output = model(prompt, **params)

                    self.logger.debug("output: %s", output["choices"][0]["text"])
                    striped_output = output["choices"][0]["text"].strip('.\n ')
                    parsed_output = []

                    # Test the different possible patterns commonly encountered:
                    if ", " in striped_output:
                        parsed_output = striped_output.split(", ")

                    elif "\n* " in striped_output or "* " in striped_output:
                        parsed_output = re.split("\* |\n\* ", striped_output)

                    elif striped_output.startswith("1. "):
                        parsed_output = re.split("\d\.\s|\n\d+\.\s", striped_output)

                    else:
                        self.logger.warning("Separator in the following output is not handled %s", striped_output)
                        continue
                    
                    if '' in parsed_output:
                        parsed_output.remove('')

                    for kw in parsed_output:
                        d[kw.lower()] += 1
                
                    model.reset()
                
                # Save to cache dir:
                self.cache_keywords[pmid] = d
            
        # Compare each keyword to the to_exclude list using Sequence Matcher. 
        # From https://docs.python.org/3/library/difflib.html#module-difflib, we will use the 0.6 as a standard parameter
        # Also check for the punctuation
        self.logger.debug("Excluding KW containing name of chemicals or organisms, or punctuations")
        kws_to_exclude = []
        for kw in d.keys():

            if any(p in kw for p in PUNCTUATION):
                self.logger.debug("Excluding %s because it contains punctuations", kw)
                kws_to_exclude.append(kw)
                continue

            comparisons = [SequenceMatcher(None, kw.lower(), w).ratio() for w in to_exclude]
            index_of_matches = np.where(np.array(comparisons) >= th_ratio)[0]
            if len(index_of_matches) > 0:
                self.logger.debug("Excluding %s because it is (similar ratio >= %.1f) to: %s", kw, th_ratio, ', '.join([to_exclude[i] for i in index_of_matches]))
                kws_to_exclude.append(kw)

        # First step of exlcusions
        for kw_to_exclude in kws_to_exclude:
            d.pop(kw_to_exclude)
        
        # Exclusion from ban-list:
        for kw in list(d.keys()):
            if any([ban_tok in kw for ban_tok in BANNED_TOKENS]):
                self.logger.debug("Excluding %s because it contains one of the banned tokens", kw)
                d.pop(kw)

        self.logger.debug("exporting the final list")
        kw = list(d.keys())
        counts = list(d.values())
        ordered_counts = np.argsort(counts)
        ordered_kw = [kw[i] for i in ordered_counts[::-1]]
        
        n = min(top_n, len(ordered_kw))

        return ordered_kw[:n]
    
    def main_findings_verbaliser(self, item:dict, n:int, chemical_classes:pd.DataFrame, compressed_labels=False, **kwargs) -> list:

        LMF = namedtuple("label_mf_pair", ["labels", "main_findings", "all_org_mentions", "all_chem_mentions"])

        original_chemical_labels = pd.DataFrame(item["chemicals"]).fillna("")

        list_of_verbalise_main_findings = []

        for i in range(n):
            
            self.logger.debug("- Running sample %d", i)

            # Get main findings 
            main_findings = create_list_of_np(item=item, chemical_classes=chemical_classes, original_chemical_labels=original_chemical_labels, logger=self.logger)

            # Shuffle
            sample = shuffle(main_findings, logger=self.logger)

            # Labels and stringly main findings
            verbalised_sample, sample_labels, all_org_mentions, all_chem_mentions = stringify(org_chem_dict=sample, original_chemical_labels=original_chemical_labels, compressed_labels=compressed_labels, logger=self.logger)

            # store
            lmf = LMF(sample_labels, verbalised_sample, all_org_mentions, all_chem_mentions)

            list_of_verbalise_main_findings.append(lmf)
        
        return list_of_verbalise_main_findings