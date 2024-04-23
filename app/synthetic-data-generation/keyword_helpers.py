# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import logging
import numpy as np
import pandas as pd
import json
import random
import re
import urllib3

from helpers import get_std_logger


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


PATHOGENS = ["Escherichia coli", "Mycobacterium tuberculosis", "Salmonella enterica", "Streptococcus pneumoniae", "Klebsiella pneumoniae", "Staphylococcus aureus", "Acinetobacter baumannii", "Neisseria gonorrhoeae", "Pseudomonas aeruginosa", "Enterococcus faecium", "Listeria monocytogenes", "Campylobacter jejuni", "Enterococcus faecalis", "Borreliella burgdorferi", "Clostridioides difficile", "Shigella sonnei", "Enterobacter hormaechei", "Helicobacter pylori", "Campylobacter coli", "Streptococcus suis", "Vibrio parahaemolyticus", "Neisseria meningitidis", "Streptococcus pyogenes", "Clostridiales bacterium", "Mycobacteroides abscessus", "Burkholderia pseudomallei", "Enterobacter cloacae", "Staphylococcus epidermidis", "Vibrio cholerae", "Bacillus cereus", "Akkermansia muciniphila", "Serratia marcescens", "Pseudomonas viridiflava", "Lactiplantibacillus plantarum", "Citrobacter freundii", "Shigella flexneri", "Stenotrophomonas maltophilia", "Bacillus thuringiensis", "Klebsiella quasipneumoniae", "Ligilactobacillus murinus", "Thermoplasmata archaeon", "Limosilactobacillus reuteri", "Bacteroides thetaiotaomicron", "Proteus mirabilis", "Legionella pneumophila", "Clostridium perfringens", "Clostridium botulinum", "Bordetella pertussis", "Rhizobium leguminosarum", "Klebsiella variicolah"]
CLASSIC_NATURAL_PRODUCTS = ["Beta-Sitosterol", "Quercetin", "Sitogluside", "Ursolic acid", "Stigmasterol", "Rutin", "Piceatannol", "Oleanolic acid", "Kaempferol", "Luteolin", "Apigenin", "Chlorogenic Acid", "Gallic Acid", "(-)-Epicatechin", "Cianidanol", "Lupeol", "Caffeic Acid", "Astragalin", "Scopoletin", "Quercitrin", "3,4-Dihydroxybenzoic acid", "Acteoside", "actinomycin D", "Betulinic acid", "Hyperoside", "4-Hydroxybenzoic acid", "4-Oxoniobenzoate", "Hexadecanoate;hydron", "Palmitic Acid", "Ferulic acid", "Chrysophanol", "beta-Amyrin", "Vanillic acid", "Cynaroside", "Isoquercitrin", "Berberine", "4-Hydroxycinnamic acid", "Paclitaxel", "Valinomicin", "Friedelin", "alpha-Amyrin", "Cosmosiin", "Emodin", "Resveratrol", "Bergapten", "Genistein", "beta-Sitosterol 3-O-beta-D-galactopyranoside", "3,4-Dihydroxy cinnamic acid", "Hydron;octadecanoate", "Stearic Acid", "1,3-Benzenediol, 5-[(1Z)-2-(4-hydroxyphenyl)ethenyl]-", "Nicotiflorin", "Ellagic acid", "Linoleic Acid", "Natamycin", "3-(4-Hydroxy-3-methoxyphenyl)prop-2-enoic acid", "Campesterol", "L-Epicatechin", "Liriodenine", "alpha-Pinene", "Caryophyllene", "Myricetin", "Methyl gallate", "Umbelliferone", "Vanillin", "Formononetin", "Oleic Acid", "Methoxsalen", "Mithramycin a", "Physcion", "Daidzein", "Myrcene", "2-Propenoic acid, 3-(4-hydroxyphenyl)-, (2Z)-", "Beta-Carotene", "Limonene", "(-)-Epigallocatechin gallate", "Vitexin", "Afzelin", "Isovitexin", "Naringenin", "Phytol", "Syringin", "Doxorubicin", "Pinoresinol", "Amentoflavone", "Kaurenoic acid", "Wogonin", "Imperatorin", "Psoralen", "3-Hydroxyolean-12-en-28-oic acid", "Auraptene", "Clavulanic acid", "Ergosterol peroxide", "Rosmarinic acid", "Betulin", "Sucrose", "4-Hydroxybenzaldehyde", "Caffeine", "2-(4-Hydroxyphenyl)ethanol", "Adenosine"]
AMINO_ACIDS = ["Alanine", "Arginine", "Asparagine", "Aspartic acid", "Cysteine", "Glutamine", "Glutamic acid", "Glycine", "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine", "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptopnam", "Tyrosine", "Valine"]
VITAMINS = ["Thiamine", "Riboflavin", "Vitamin B3", "Vitamin B5", "Pyridoxine", "Biotin", "Folate", "Vitamin B12", "Vitamin C", "Vitamin K"]
OTHER_NUTRITIENTS = ["Glucose", "Fructose", "Sucrose", "Lactose", "Cellulose", "Glycogen", "peptones"]
MINERALS = ["Azote", "Potassiun", "Bore", "sodium ", "calcium", "Magnesium", "Zinc", "Fer"]

MEDIUM_CULTURE_CHEMICAL_KEYPHRASES = ["substrates include {chemicals}", "incubated with {chemicals}", "the culture medium contains {chemicals}"]
COMPARISON_CHEMICAL_KEYPHRASES = ["similar structure to {chemicals}", "similar activity to {chemicals}", "higher activity than {chemicals}", "lower activity than {chemicals}", "same profile as {chemicals}"]
ACTIVITY_ON_ORGANISM = ["antibacterial activity against {organisms}"]


class AlternativeKeyphraseSampler():
    def __init__(self, logger_level:int = logging.DEBUG, log_dir:str = ".", **kwargs):

        # list of items
        self.logger = kwargs["logger"] if "logger" in kwargs else get_std_logger(name="Alernative-keyphrases-sampler", level=logger_level, path=log_dir)
        self.PATHOGENS = PATHOGENS
        self.NP = CLASSIC_NATURAL_PRODUCTS
        self.AMINO_ACIDS = AMINO_ACIDS
        self.VITAMINS = VITAMINS
        self.OTHER_NUTRITIENTS = OTHER_NUTRITIENTS
        self.MINERALS = MINERALS
        self.ALL_CHEMICALS = self.NP + self.AMINO_ACIDS + self.VITAMINS + self.OTHER_NUTRITIENTS + self.MINERALS
        self.CHEMICAL_SET_SIZES = [len(self.NP), len(self.AMINO_ACIDS), len(self.VITAMINS), len(self.OTHER_NUTRITIENTS), len(self.MINERALS)]

        # list of keyphrases to fill
        self.MEDIUM_CULTURE_CHEMICAL_KEYPHRASES = MEDIUM_CULTURE_CHEMICAL_KEYPHRASES
        self.COMPARISON_CHEMICAL_KEYPHRASES = COMPARISON_CHEMICAL_KEYPHRASES
        self.ACTIVITY_ON_ORGANISM = ACTIVITY_ON_ORGANISM

    def join_names(self, list_of_names):
        if len(list_of_names) > 1:
            verbalised = ", ".join(list_of_names[:-1]) + " and " + list_of_names[-1]
        else:
            verbalised = list_of_names[0]

        return verbalised
    
    def get_a_medium_culture_keyphrase(self, max_chem:int = 3):
        template = random.sample(self.MEDIUM_CULTURE_CHEMICAL_KEYPHRASES, 1)[0]
        n_chemical = random.sample(range(1, max_chem+1), 1)[0]
        chemicals = np.random.choice(a=self.ALL_CHEMICALS, size=n_chemical, replace=False, p=np.repeat([1/len(self.CHEMICAL_SET_SIZES) * 1/l for l in self.CHEMICAL_SET_SIZES], self.CHEMICAL_SET_SIZES, axis=0))
        medium_keyphrase = template.format(chemicals=self.join_names(chemicals))
        
        return medium_keyphrase
    
    def get_a_chemical_comparison_keyphrase(self, max_chem:int = 3):
        template = random.sample(self.COMPARISON_CHEMICAL_KEYPHRASES, 1)[0]
        n_chemical = random.sample(range(1, max_chem+1), 1)[0]
        chemicals = random.sample(self.NP, n_chemical)
        comparison_keyphrase = template.format(chemicals=self.join_names(chemicals))
        
        return comparison_keyphrase

    def get_an_activity_on_organism(self, max_org:int = 5):
        template = random.sample(self.ACTIVITY_ON_ORGANISM, 1)[0]
        n_organisms = random.sample(range(1, max_org+1), 1)[0]
        orgamisms = random.sample(self.PATHOGENS, n_organisms)
        activity_keyphrase = template.format(organisms=self.join_names(orgamisms))
        
        return activity_keyphrase
    
    def get_proba_from_expected_proportion(self, p):
        """Will give the probability such that we can expect a proportion p of them WIHTOUT modif"""
        return 1 - np.exp((np.log(p)/3))
    
    def modify_keywords(self, original_keywords, proba):
        
        def apply_modif(keywords, f, args):
            if len(keywords) <= 5:
                # We ADD new keywords
                keywords.append(f(**args))
            else:
                # We replace existing keywords:
                random_index = random.randint(0, len(keywords) - 1)
                _removed_element = keywords.pop(random_index)
                _new_elements = f(**args)
                self.logger.info("Replace %s by %s", _removed_element, _new_elements)
                keywords.insert(random_index, _new_elements)
        
        working_keywords = original_keywords.copy()
        for (f , args) in [(self.get_a_medium_culture_keyphrase, {}), (self.get_a_chemical_comparison_keyphrase, {}), (self.get_an_activity_on_organism, {})]:
            if np.random.uniform() <= proba:
                apply_modif(working_keywords, f , args)
        
        return working_keywords