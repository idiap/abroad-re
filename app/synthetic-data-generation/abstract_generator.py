# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import logging
import numpy as np
from llama_cpp import Llama
from abc import ABC, abstractmethod
from helpers import get_std_logger


class SyntheticData(ABC):

    @abstractmethod
    def export(self, *args, **kwargs):
        pass


class LotusSyntheticData(SyntheticData):

    def __init__(self, id, pmid, prompt, title, abstract, organisms, chemicals, relations, generation_params):
        self.id = id
        self.pmid = pmid
        self.prompt = prompt
        self.title = title
        self.abstract = abstract
        self.organisms = organisms
        self.chemicals = chemicals
        self.relations = relations
        self.generation_params = generation_params
        self.selector_output = {}

    def export(self, *args, **kwargs) -> dict:

        item = {}
        item["ArticleTitle"] = self.title
        item["prompt"] = self.prompt
        item["parameters"] = self.generation_params
        item["AbstractText"] = self.abstract
        item["score"] = self.selector_output["score"]
        item["chemicals"] = [{"id": c["id"], "label": c["label"], "type": c["type"]} for c in self.chemicals]
        item["organisms"] = self.organisms
        item["relations"] = self.relations

        return item
    
    def post_processing(self, *args, **kwargs):
        self.abstract = self.abstract.strip(" \n")

        # To alleviate the high frequencx if abstract starting by "In this study, ..."
        if self.abstract.startswith("In this study, ") and not self.abstract.startswith("In this study, we"):
            self.abstract = self.abstract.replace("In this study, ", "")
            self.abstract = self.abstract.capitalize()



class AbstractGenerator(ABC):

    @abstractmethod
    def generate_abstract(self, prompt, *args, **kwargs):
        pass

class LLamaAbstractGenerator(AbstractGenerator):

    def __init__(self, model_path, model_params, logger_name="LLama-abstract-generator", logger_level=logging.DEBUG, log_dir = ".", **kwargs):
        self.llm_model_path = model_path
        self.llm_model_params = model_params
        self.llm_model = None
        self.logger = kwargs["logger"] if "logger" in kwargs else get_std_logger(name=logger_name, level=logger_level, path=log_dir)

    def load_model(self, verbose=False):
        try:
            self.llm_model = Llama(model_path=self.llm_model_path, **self.llm_model_params)
            self.llm_model.verbose = False
        except ValueError as e_init:
            self.logger.error(str(e_init))
            sys.exit(1)
    
    def generate_abstract(self, prompt, parameters):

        if self.llm_model is None:
            self.logger.info("Loading model from %s", self.llm_model_path)
            self.load_model()
        
        try:
            self.llm_model.reset()
            output = self.llm_model(prompt, **parameters)
            generated_abstract = output["choices"][0]["text"]
            return generated_abstract

        except ValueError as e1:
            self.logger.error("Error while generating abstract: %s", str(e1))
            return False

        except RuntimeError as e2:
            self.logger.error("Error while generating abstract: %s", str(e2))
            return False

        except Exception as ef:
            self.logger.error("An unexpected error occurred while generating abstract: %s", str(ef))
            return False


class AbstractSelector(ABC):

    @abstractmethod
    def select(self, list_of_synthetic_abstracts:list, n:int, *args, **kwargs) -> list:
        pass

class LotusAbstractSelector(AbstractSelector):

    def __init__(self, logger_name="abstract-selector", logger_level=logging.DEBUG, log_dir=".", **kwargs):
        self.logger = kwargs["logger"] if "logger" in kwargs else get_std_logger(name=logger_name, level=logger_level, path=log_dir)

    def get_score(self, synthetic_data:LotusSyntheticData):
        
        lowered_abstract = synthetic_data.abstract.lower()

        chemicals_in_abstract = []
        organisms_in_abstract = []

        prop = 0

        # Find organisms labels in asbtract and add corresponding labels
        for org_item in synthetic_data.organisms:
            org_id_l = org_item["id"]
            org_label = org_item["label"].lower()
            splited_org_label = org_label.split()
            if len(splited_org_label) > 1:
                if org_label in lowered_abstract or splited_org_label[0][0] + '. ' + splited_org_label[1] in lowered_abstract:
                    organisms_in_abstract.append(org_id_l)
            else:
                if org_label in lowered_abstract:
                    organisms_in_abstract.append(org_id_l)
        
        # Find chemicals labels in asbtract and add corresponding labels
        for chem_item in synthetic_data.chemicals:
            chem_id_l = chem_item["id"]
            chem_label = chem_item["prompt_label"].lower()
            if chem_label in lowered_abstract:
                chemicals_in_abstract.append(chem_id_l)

        # What proportion of relations have both the head and tail of the relation in the abstract ??
        for r in synthetic_data.relations:
            if (r[0] in organisms_in_abstract) and (r[1] in chemicals_in_abstract):
                prop += 1
        
        score = prop / len(synthetic_data.relations)

        return score

    def select(self, list_of_synthetic_data:list, n:int, score_th:float, *args, **kwargs):
        score_list = []
        final_synthetic_data_list = []
        
        for synthetic_data in list_of_synthetic_data:
            
            data_score = self.get_score(synthetic_data)
            self.logger.info("ID %s: score %.2f", synthetic_data.id, data_score)     

            if data_score == 0:
                self.logger.warning("ID %s: the generated abstract was discarded because it either don't contain any organisms or chemicals.\nAbstract: %s",  synthetic_data.id, synthetic_data.abstract)
                continue
            
            if data_score < score_th:
                self.logger.warning("ID %s: the generated abstract was discarded because it does not meet the score threshold: %.2f < %.2f", synthetic_data.id, data_score, score_th)
                continue
        
            final_synthetic_data_list.append(synthetic_data)
            score_list.append(data_score)
        
        if not len(final_synthetic_data_list):
            return []

        assert len(final_synthetic_data_list) == len(score_list)

        ordered_score = np.argsort(score_list)[::-1]

        # Select top-n
        top_n = min(len(ordered_score), n)
        self.logger.info("Export the top %d", top_n)

        output = []
        for i in ordered_score[:top_n]:
            o = final_synthetic_data_list[i]
            o.selector_output["score"] = score_list[i]
            output.append(o)

        return output
