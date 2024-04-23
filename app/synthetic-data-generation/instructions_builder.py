# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import logging
import json
import re
import numpy as np
import pandas as pd

from enum import Enum
from collections import namedtuple
from abc import ABC, abstractmethod
from collections import defaultdict
from difflib import SequenceMatcher
from instructions_helpers import create_list_of_np, shuffle, stringify
from helpers import get_std_logger



class InstructionsBuilder(ABC):
    @abstractmethod
    def main_findings_verbaliser(self, item:dict, *args, **kwars) -> list:
        pass

class KeywordsExtractor(ABC):
    @abstractmethod
    def extract(self, item:dict, *args, **kwars) -> dict:
        pass

class KeywordPrompts(Enum):
    DEFAULT = (f"Extract a comma-separated list of keywords from the following abstract of a scientific article.\n"
                "Input: {text}\n"
                "Keywords:")
    MISTRAL = (f"<s>[INST]Task: Extract a comma-separated list of keywords from the following title and abstract of a scientific article.\n"
                "Input: {text}\n"
                "Keywords:[/INST]")

class InstructionsPrompts(Enum):
    DEFAULT = (f"Instructions: Given a title, a list of keywords and main findings, create an abstract for a scientific article.\n"
            "Title: {title}\n"
            "Keywords: {keywords}\n"
            "Main findings: {main_findings}\n"
            "Abstract:")
    VICUNA = (f"A chat between a curious user and an artificial intelligence assistant. "
            "The assistant helps write abstracts of scientific articles in biology. "
            "The assistant must not start the created abstract with common formulations such as 'In this study', 'This article reports', 'The present study', or something similar. "
            "USER: Given a title, a list of keywords and main findings, create an abstract for a scientific article. "
            "Title: {title} "
            "Keywords: {keywords} "
            "Main findings: {main_findings} "
            "ASSISTANT: \nAbstract:")
    MISTRAL = ("<s>[INST] Given a title, a list of keywords and main findings, create an abstract for a scientific article.\n"
            "Title: {title}\n"
            "Keywords: {keywords}\n"
            "Main findings: {main_findings}\n"
            "Notes: Avoid starting the abstract with generic formulations such as 'In this study', 'This article reports', 'The present study', etc.\n"
            "Abstract: [/INST]")

class LotusKeywordsExtractor(KeywordsExtractor):
    def __init__(self, model, logger_level=logging.DEBUG, log_dir=".", **kwargs) -> None:
        self.model = model
        self.prompt = None
        self.DEFAULT_KWS = ["structures", "spectroscopic methods", "antibacterial activity", "structure elucidation", "new compounds", "gc-ms", "x-ray crystallography", "ic50 values", "nmr spectroscopy", "mass spectrometry"]
        self.cache_keywords = {}
        self.logger = kwargs["logger"] if "logger" in kwargs else get_std_logger(name="keywords-extractor", level=logger_level, path=log_dir)
        self.select_prompt()

    def select_prompt(self):
        if "mistral" in self.model.metadata["general.name"]:
            self.prompt = KeywordPrompts.MISTRAL.value
        else:
            self.warning("model name %s does not correspond to any prompt template.", self.model.metadata["general.name"])
            self.prompt = KeywordPrompts.DEFAULT.value

    def apply_exclusion(self, d:dict, to_exclude:list, **kwargs):
        TOP_N = 10
        DEFAULT_TH_RATIO = 0.5
        BANNED_TOKENS = ["keyword", "\n", "\t", "abstract", "scientific article"] 
        PUNCTUATION = ['!', '"', '#', '$', '&', '\', ''', '*', '+', ',', '.', '/', ':', ';', '.', '?', '^', '_', '`', '{', '|', '}', '~']

        th_ratio = kwargs.pop("th_ratio", DEFAULT_TH_RATIO)
        # Compare each keyword to the to_exclude list using Sequence Matcher. 
        # From https://docs.python.org/3/library/difflib.html#module-difflib, we will use the 0.6 as a standard parameter
        # Also check for the punctuation
        self.logger.info("Excluding KW containing name of chemicals or organisms, or punctuations")
        kws_to_exclude = []
        
        top_n = kwargs.pop("top_n", TOP_N)
        
        for kw in d.keys():

            if any(p in kw for p in PUNCTUATION):
                self.logger.info("Excluding %s because it contains punctuations", kw)
                kws_to_exclude.append(kw)
                continue

            comparisons = [SequenceMatcher(None, kw.lower(), w).ratio() for w in to_exclude]
            index_of_matches = np.where(np.array(comparisons) >= th_ratio)[0]
            if len(index_of_matches) > 0:
                self.logger.info("Excluding %s because it is (similar ratio >= %.1f) to: %s", kw, th_ratio, ', '.join([to_exclude[i] for i in index_of_matches]))
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
        
    def extract(self, item:dict, **kwargs): 

        # A list of banned token that should be removed if they are, or are contained in one keyword.
        
        GENERATION_PARAMS = {"max_tokens": 500, "repeat_penalty": 1.1, "temperature": 0, "stop": ["\n\n"]}
        DEFAULT_TEMPERATURE_LIST = [0, 0.3, 0.6, 0.9]

        # CHECKING AND INIT
        title = item["ArticleTitle"]
        abstract = item["AbstractText"]
        doc = title + " " + abstract
        pmid = item["PMID"]

        # Init from args or defaults
        d = defaultdict(lambda: 0)
    
        # Check in cache:
        if pmid in self.cache_keywords and len(self.cache_keywords[pmid]):
            self.logger.info("PMID %s: retrieve list of keywords from cache.", pmid)
            d = self.cache_keywords[pmid]
            return d

        # If not in cache, continue
        self.logger.info("PMID %s not found in cache.", pmid)
        input = self.prompt.format(text=doc)

        temperature_list = kwargs.pop("temperature_list", DEFAULT_TEMPERATURE_LIST)

        # On all temperatures 
        for temperature in temperature_list:

            # generate a random temperature from the  temperature_range params for the sample
            GENERATION_PARAMS["temperature"] = temperature
            self.logger.info("temp: %.2f", GENERATION_PARAMS["temperature"])

            # Send the prompt
            self.logger.debug("Prompt: %s", input)
            output = self.model(input, **GENERATION_PARAMS)

            self.logger.info("keywords: %s", output["choices"][0]["text"])
            striped_output = output["choices"][0]["text"].strip('.\n ')
            parsed_output = []

            # Remove potential quote ?
            if '"' in striped_output:
                striped_output = striped_output.replace('"', '')

            # Test the different possible patterns commonly encountered:
            if striped_output.startswith("1. "):
                parsed_output = re.split("\d+\.\s|\n\d+\.\s|\,\d+\.\s|\,\s\d+\.\s", striped_output)

            elif "\n* " in striped_output or "* " in striped_output:
                parsed_output = re.split("\* |\n\* ", striped_output)

            elif ", " in striped_output:
                parsed_output = striped_output.split(", ")
            
            else:
                self.logger.warning("Separator in the following output is not handled %s", striped_output)
                continue
            
            if '' in parsed_output:
                parsed_output.remove('')

            for kw in parsed_output:
                # If its not a 1 char element:
                if len(kw) > 1:
                    d[kw.lower()] += 1
        
            self.model.reset()
        
        # Save to cache dir:
        self.cache_keywords[pmid] = d

        return d

    def load_kw_cache(self, path_to_cache_file):

        if os.path.exists(path_to_cache_file):
            self.logger.info("Open cache file at %s", path_to_cache_file)
            with open(path_to_cache_file, "r", encoding="utf-8") as f_cache:
                self.cache_keywords = json.load(f_cache)
        else:
            self.logger.info("No cache file found.")
            self.cache_keywords = {}
    
    def export_keywords_cache(self, path_to_cache_file):
        
        self.logger.info("Export cache keywords")
        with open(path_to_cache_file, "w", encoding="utf-8") as f_cache_write:
            json.dump(self.cache_keywords, f_cache_write, indent=4)


class LotusInstructionsBuilder(InstructionsBuilder):

    def __init__(self, log_file_name="instruction-builder", logger_level=logging.DEBUG, log_dir=".", p_chem_class:float = 0.2, p_contract:float = 0.9, p_numbering:float = 0.25, compressed_labels = False, **kwargs):
        self.logger = kwargs["logger"] if "logger" in kwargs else get_std_logger(name=log_file_name, level=logger_level, path=log_dir)
        self.p_chem_class = p_chem_class
        self.p_contract = p_contract
        self.p_numbering = p_numbering
        self.compressed_labels = compressed_labels

    
    def select_generation_template(self, name):
        ll_names = [p.name for p in InstructionsPrompts]
        if not name in ll_names:
            self.logger.warning("Please use a name from the list of available prompts: %s. Return DEFAULT", ', '.join(ll_names))
            return InstructionsPrompts.DEFAULT.value
        else:
            return InstructionsPrompts[name].value

    def main_findings_verbaliser(self, item:dict, n:int, chemical_classes:pd.DataFrame, **kwargs) -> list:

        LMF = namedtuple("label_mf_pair", ["labels", "main_findings", "all_org_mentions", "all_chem_mentions"])

        original_chemical_labels = pd.DataFrame(item["chemicals"]).fillna("")

        list_of_verbalise_main_findings = []

        for i in range(n):
            
            self.logger.debug("- Running sample %d", i)

            # Get main findings 
            main_findings = create_list_of_np(item=item, p_chem_class=self.p_chem_class, chemical_classes=chemical_classes, original_chemical_labels=original_chemical_labels, logger=self.logger)

            # Shuffle
            sample = shuffle(main_findings, logger=self.logger)

            # Labels and stringly main findings
            verbalised_sample, sample_labels, all_org_mentions, all_chem_mentions = stringify(org_chem_dict=sample, original_chemical_labels=original_chemical_labels, p_contract=self.p_contract, p_numbering=self.p_numbering, compressed_labels=self.compressed_labels, logger=self.logger)

            # store
            lmf = LMF(sample_labels, verbalised_sample, all_org_mentions, all_chem_mentions)

            list_of_verbalise_main_findings.append(lmf)
        
        return list_of_verbalise_main_findings