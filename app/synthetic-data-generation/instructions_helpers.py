# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import random
import re
from collections import defaultdict
from copy import deepcopy
from string import ascii_lowercase, ascii_uppercase
from num2words import num2words

import more_itertools
import numpy as np
import pandas as pd


def create_list_of_np(item:dict, original_chemical_labels:pd.DataFrame, chemical_classes:pd.DataFrame, p_chem_class:float, **kwargs):
    """
    Create a list of relations between organisms and natural products.
    
    - Each element in the list is a dict containing all the relations between 1 organisms and its N associated chemicals: {'org_id': 'Q...', 'org_label': 'Genus species', 'chemicals': {a dict of chemicals}}
    - In the dict of chemicals attached to an organism, the key is the main name of the chemical, or in the case of compounds derivates (e.g hansfordiol A-I), its longest prefix (e.g hansfordiol)
    - In the case of a single chemical, the dict looks like: {'Marmesin': {'id': 'Q13847605', 'type': 'chemical', 'suffixes': None} 
    - In the case of compound with several derivates, the suffixes that identifiy the derivates are stored in a list and the id is attached to the suffix: {'hansfordiol': {'id': None, 'type': 'chemical', 'suffixes': [[('A', 'Q110170266'), ('B', 'Q110170267'), ... , ('I', 'Q110170274'), ('J', 'Q110170275')]]} 
    - Consecutive list of suffixes are later grouped in sub-lists.

    

    Args:
        item (dict): a literature item from the LOTUS train/valid set
        chemical_classes (pd.DataFrame, optional): a DataFrame containing all the available chemical classes annotations. Each compound should only have one annotation. The standard table contains 3 columns: structure_wikidata, structure_taxonomy_npclassifier_02superclass, class_id
        p_chem_class (float): the probability of replacing at least 2 chemicals from the same class and produced by the same organism to their corresponding class.
    
    Returns: list of relations between organisms and natural products. 
    """
    def get_ordering_function(type):
        """Get an ordering function for different types of suffixes in order to determine consecutive lists

        Args:
            type (str): can be "alpha" for lower alphabeticals, "ALPHA" for upper alphabeticals, "num" for numbers, "alphanum" for lower alphanumeric, "ALPHANUM" for upper alphanumeric and "roman" for roman numbers.

        Returns:
            function: an ordering function
        """
        
        ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]
        ALPHANUMLOWER = [a + str(n) for a in ascii_lowercase[0:26] for n in range(0,101)]
        ALPHANUMUPPER = [a + str(n) for a in ascii_uppercase[0:26] for n in range(0,101)]

        if type == "alpha":
            return ascii_lowercase.index
        if type == "ALPHA":
            return ascii_uppercase.index
        if type == "num":
            return lambda x: int(x)
        if type == "roman":
            return lambda x: ROMAN.index(x)
        if type == "alphanum":
            return lambda x: ALPHANUMLOWER.index(x)
        if type == "ALPHANUM":
            return lambda x: ALPHANUMUPPER.index(x)
    
    def order(seq, ordering):
        """From a sequence of suffixes and a dedicated ordering function, return the ordered sequence

        Args:
            seq (list): a sequence of suffixes
            ordering (function): an ordering function 

        Returns:
            list: the ordered sequence
        """
        indexes = [ordering(x) for x in seq]
        ordered_indexes = np.argsort(indexes)
        return [seq[i] for i in ordered_indexes]

    MATCHING_DICT_REGEX = {
        "alpha": "^[a-z]$",
        "ALPHA": "^[A-Z]$",
        "num": "^[0-9]+$",
        "roman": "X?(X|IX|IV|V?I{0,3})?$",
        "alphanum": "^[a-z][0-9]*$",
        "ALPHANUM": "^[A-Z][0-9]*$",
    }
    def check_romans(d_contract):
        """
        Check that roman numbers like 'I' or 'V' were not confound with ALPHA suffixes.
        """
        for i in range(len(d_contract)):
            for chem in d_contract[i]["chemicals"]:
                if d_contract[i]["chemicals"][chem]["suffixes"] is not None:
                    if len(d_contract[i]["chemicals"][chem]["suffixes"]["roman"]) > 0:
                        for j in [index for index in range(len(d_contract[i]["chemicals"][chem]["suffixes"]["ALPHA"])) if d_contract[i]["chemicals"][chem]["suffixes"]["ALPHA"][index][0] in ["I", "V", "X"]][::-1]:
                            _s = d_contract[i]["chemicals"][chem]["suffixes"]["ALPHA"].pop(j)
                            d_contract[i]["chemicals"][chem]["suffixes"]["roman"].append(_s)
                            
    # logger, chem-class proba and intial tables
    logger = kwargs.pop("logger", logging.getLogger())

    chemicals_df = pd.DataFrame(item["chemicals"]).fillna("")
    organisms_df = pd.DataFrame(item["organisms"]).fillna("")
    relations_df = pd.DataFrame(item["relations"], columns=["organism_wikidata", "structure_wikidata"])

    # Init 
    relations_df_classes = pd.DataFrame()
    d_chemicals = []

    if chemical_classes is not None:
        # Map the chemical classes to the relation df:
        relations_df = relations_df.merge(chemical_classes[chemical_classes["structure_wikidata"].isin(relations_df["structure_wikidata"].tolist())], how="left", on="structure_wikidata")

        # Extract the number of times each class is associated with one organism. We don't want to integrate a chemical classes that only bring one chemical for an organism
        counts_per_organisms = relations_df.dropna().groupby(['organism_wikidata', 'class_id']).size().to_frame(name = 'count').reset_index()
        
        for i, r in counts_per_organisms.iterrows():

            # If there is at least 2 chemical of this class related to the organism, it can be replaced by the class
            if r["count"] > 1 and random.uniform(0, 1) <= p_chem_class:

                logger.debug("The %d relations between organism id %s and chemicals from the class %s have been replaced by a direct association to the class", r["count"], r["organism_wikidata"], r["class_id"])

                # Extract the associated relations and add them to the relations_df_classes dataframe 
                sub_df = relations_df[(relations_df["organism_wikidata"] == r["organism_wikidata"]) & (relations_df["class_id"] == r["class_id"])]
                relations_df_classes = pd.concat([relations_df_classes, sub_df]).reset_index(drop=True)

                # Remoe them from relations_df
                relations_df.drop(sub_df.index, inplace=True)

    # Prepare the list
    for org_id, chems_ids in relations_df.groupby('organism_wikidata'):
        org_label = organisms_df[organisms_df["id"] == org_id]["label"].values[0]
        sub = {"org_id": org_id, "org_label": org_label, "chemicals": []}
        for chem_id in chems_ids["structure_wikidata"].tolist():
            chem_label = chemicals_df[chemicals_df["id"] == chem_id]["label"].values[0]
            sub["chemicals"].append({"chem_id": chem_id, "chem_label": chem_label})
        d_chemicals.append(sub)
    
    # 2 - contract names
    d_contract = []
    sub_template = {"org_id": "", "org_label": "", "chemicals": defaultdict(lambda: {"id": None, "type": "chemical", "suffixes": {"roman": [], "alpha":[], "ALPHA":[], "num": [], "alphanum":[], "ALPHANUM":[]}})}

    # If there are chemicals
    for org_item in d_chemicals:
        
        sub = deepcopy(sub_template)

        sub["org_id"] = org_item["org_id"]
        sub["org_label"] = org_item["org_label"]

        for chem_items in org_item["chemicals"]:

            chem_label = chem_items["chem_label"]
            chem_id = chem_items["chem_id"]

            # Split by space or '-'. The last is considered as the suffix
            split_string = re.split(r'[\s-]', chem_label)

            # When no suffixes, the key is the complete name and the attribute id is directly linked to
            if len(split_string) == 1:
                logger.debug("No suffixes found for %s", chem_label)
                sub["chemicals"][chem_label]["id"] = chem_id
                sub["chemicals"][chem_label]["suffixes"] = None

            else:
                suffix = split_string[-1]
                matched_suffix = False

                # Test the suffix against all regex in MATCHING_DICT_REGEX to determine its type:
                for type, r in MATCHING_DICT_REGEX.items():
                    if re.match(r, suffix):
                        logger.debug("The suffix %s of %s matched for a derivate was determined to be %s.", suffix, chem_label, type)
                        pos = chem_label.rfind(suffix)
                        prefix = chem_label[:(pos - 1)]
                        # Prefix is lowered in there are some inconsistencies in the names.
                        # The key is the predix and the id is associated to the suffix
                        sub["chemicals"][prefix.lower()]["suffixes"][type].append((suffix, chem_id))
                        matched_suffix = True
                        break

                # if the suffix dosen't match because it is 'acid' for instance, add like a regular item
                if not matched_suffix:
                    logger.debug("A suffix was found for %s, but did not match any of the categories (roman, ALPHA, num, etc.)", chem_label)
                    sub["chemicals"][chem_label]["id"] = chem_id
                    sub["chemicals"][chem_label]["suffixes"] = None

        d_contract.append(sub)

    
    # check for roman vs alpha confusions
    check_romans(d_contract)

    # A list containing tuple (org_label, chem_prefix, suffix). Each element in this list has only one suffix. Then, it should be converted back to a standard chem
    to_transform = []
    
    # Ordering and find consecutive list:
    for i in range(len(d_contract)):

        for chem_prefix in d_contract[i]["chemicals"]:

            # If some prefix were found:
            if d_contract[i]["chemicals"][chem_prefix]["suffixes"] is not None:

                # If there is only one suffix found, just rebuild the original name by joining with a space, no need of the sequences
                suffixes = [s for l in d_contract[i]["chemicals"][chem_prefix]["suffixes"].values() for s in l]
                if len(suffixes) == 1:
                    logger.debug("Only one suffix (%s) for %s. Prepare to re-merge", suffixes[0], chem_prefix)
                    to_transform.append((i, chem_prefix, suffixes[0]))
                    continue

                # If there were more than 1, try to found consecutive sequences:
                for type, l in d_contract[i]["chemicals"][chem_prefix]["suffixes"].items():
                    logger.debug("More than one suffix found for %s, look for consecutive lists", chem_prefix)
                    ordering = get_ordering_function(type)
                    l_suffixes = [s[0] for s in l]
                    _consecutives = [list(g) for g in more_itertools.consecutive_groups(order(l_suffixes, ordering), ordering)]
                    consecutives = []
                    for cons_s in _consecutives:
                        consecutive_list = [l[l_suffixes.index(s)] for s in cons_s]
                        logger.debug("Consecutive list of suffixes found: %s", ', '.join([s[0] for s in consecutive_list]))
                        consecutives.append(consecutive_list)
                    d_contract[i]["chemicals"][chem_prefix]["suffixes"][type] = consecutives
                
                # Then merge as a single list:
                d_contract[i]["chemicals"][chem_prefix]["suffixes"] = [l for ll in d_contract[i]["chemicals"][chem_prefix]["suffixes"].values() for l in ll]


    # Remove old version when only one suffix was found
    for org_i, chem_prefix, suffix in to_transform:
        d_contract[org_i]["chemicals"].pop(chem_prefix)
        chem_id = suffix[1]
        original_chem_label = original_chemical_labels[original_chemical_labels["id"] == chem_id]["label"].values[0]
        d_contract[org_i]["chemicals"][original_chem_label]["suffixes"] = None
        d_contract[org_i]["chemicals"][original_chem_label]["id"] = chem_id

    # Add chemical classes (if any):
    if not relations_df_classes.empty:

        # Browse by org_id and class_id and add to the list
        for org_class_ids, data in relations_df_classes.groupby(['organism_wikidata', 'class_id']):
            
            # Get all the currently added organism (can change during iteration)
            org_id_list = [org_item["org_id"] for org_item in d_contract]

            org_id = org_class_ids[0]
            class_id = org_class_ids[1]
            class_label = chemical_classes[chemical_classes["class_id"] == class_id]["structure_taxonomy_npclassifier_02superclass"].values[0]
            list_of_instances = data["structure_wikidata"].tolist()
            n = len(list_of_instances)
            
            # If some chemicals were already added to this organism:
            if org_id in org_id_list:
                i = org_id_list.index(org_id)
                d_contract[i]["chemicals"][class_label] = {"id": class_id, "type": "class", "n": n, "instances":list_of_instances, "suffixes": None}
            
            # If not we need to add it.
            else:
                org_label = organisms_df[organisms_df["id"] == org_id]["label"].values[0]
                d_contract.append({"org_id": org_id, "org_label": org_label, "chemicals": {class_label: {"id": class_id, "type": "class", "n": n, "instances":list_of_instances, "suffixes": None}}})

    return d_contract

def shuffle(input, **kwargs):
    """Shuffle the dict returned by create_list_of_np by organisms and chemical. From the new order of the entities over iteration, a rank is also associated.

    Args:
        input (list): the list of relations from create_list_of_np

    Returns:
        list: the shuffled list with numbering of entities
    """

    # Copy the input dict
    org_chem_dict = deepcopy(input)
    logger = kwargs.pop("logger", logging.getLogger())

    # Shuffeling
    logger.debug("Shuffling list of pairs")
    ## Shuffle chems associated to each org in the input dict
    for i in range(len(org_chem_dict)):
        shuffled = list(org_chem_dict[i]["chemicals"].items())
        random.shuffle(shuffled)
        org_chem_dict[i]["chemicals"] = dict(shuffled)
    
    ## Shuffle orgs
    random.shuffle(org_chem_dict)

    # Get the whole set of chemicals for numbering:
    all_chem = list([(i, p, val) for i in range(len(org_chem_dict)) for (p, val) in org_chem_dict[i]["chemicals"].items()])

    # Create the ranking
    logger.debug("Add a numbering to the chemicals")
    dict_ranker = {}
    counter = 1
    for chem in all_chem:
        chem_label = chem[1]
        chem_type = chem[2]["type"]
        suffixes = chem[2]["suffixes"]
        # If suffixes is None, it is either a class or a single chemical wihout suffixes
        if suffixes is None:
            # If it already exists in the dict, add the info of its rank, else, add it and increment the rank
            if chem_label in dict_ranker:
                org_chem_dict[chem[0]]["chemicals"][chem_label]["suffixes"] = (None, dict_ranker[chem_label])
            else:
                dict_ranker[chem_label] = counter
                org_chem_dict[chem[0]]["chemicals"][chem_label]["suffixes"] = (None, counter)
                
                # If it is a class, we should add the total number of involved entities for increasing the count, because there are implicitly n chemicals
                c = 1 if chem_type == "chemical" else chem[2]["n"]
                counter += c
        # If there are suffixes, it means that it correspond to a derivates (e.g Flavone A-D)
        else:
            # All combinations of prefix + suffix are tested
            for l_s_i in range(len(suffixes)):
                for s_i in range(len(suffixes[l_s_i])):
                    if (chem_label + suffixes[l_s_i][s_i][0]) in dict_ranker:
                        org_chem_dict[chem[0]]["chemicals"][chem_label]["suffixes"][l_s_i][s_i] = (*suffixes[l_s_i][s_i], dict_ranker[(chem_label + suffixes[l_s_i][s_i][0])])
                    else:
                        dict_ranker[(chem_label + suffixes[l_s_i][s_i][0])] = counter
                        org_chem_dict[chem[0]]["chemicals"][chem_label]["suffixes"][l_s_i][s_i] = (*suffixes[l_s_i][s_i], counter)
                        counter += 1

    return org_chem_dict

def get_relations_labels(org_chem_dict, **kwargs):
    """
    From a list of relations between organisms and chemicals, return the corresponding list of pairs of IDs to create the labels to the abstract.

    Args:
        org_chem_dict (list): list of relations between organisms and chemicals

    Returns:
        _type_: the list of relations: [[id_org_1, id_chem_1], [id_org_1, id_chem_2], [id_org_2, id_chem_3], ...]
    """

    logger = kwargs.pop("logger", logging.getLogger())

    logger.debug("Extract pairs of relations for labels")
    relations = []
    for org_item in org_chem_dict:
        org_id = org_item["org_id"]
        for chems_items in org_item["chemicals"].values():
            if chems_items["id"] is not None:
                relations.append([org_id, chems_items["id"]])
            else:
                for chems_suffixes_items in [item for l_items in chems_items["suffixes"] for item in l_items]:
                    relations.append([org_id, chems_suffixes_items[1]])
    return relations



def stringify(org_chem_dict, original_chemical_labels, p_contract:float, p_numbering:float, compressed_labels:bool, **kwargs):
    """
    Stringify a list of relation between organisms and compounds. 
    Several alternatives ways of writing will be randomly selected and combined: numbering of compounds, compression of derivates, sentence templates. 

    Args:
        org_chem_dict (list): the list of relations, shuffled from shuffle.
        original_chemical_labels (pd.DataFrane) the original table with the chemical names.
        p_contract (float, optional): probability of contracting a list of derivates. Defaults to 0.9.
        p_numbering (float, optional): probability of numbering the compounds. Defaults to 0.25.

    Returns:
        _type_: _description_
    """

    # VARS
    CONTRACT_FORM = [("{start}-{end}", 1)] # , ("{start} to {end}", 0.1)
    MENTION_FORM = [("{chems} were isolated from {org}", 0.9), ("{org} produces {chems}", 0.1)]
    logger = kwargs.pop("logger", logging.getLogger())

    logger.debug("Start verbalisation of pairs.")
    # Are we gonna number the chemicals ?
    numbering = random.uniform(0, 1) <= p_numbering
    logger.debug("numbering of natural products: %s", numbering)

    # List of mentions for checkings
    all_organisms_mentions = []
    all_chemicals_mentions = {}
    all_relations = []

    # Final list of mentions
    mention_list = []
    
    for org_item in org_chem_dict:
        
        org_label = org_item["org_label"]
        org_id = org_item["org_id"]
        all_organisms_mentions.append({"id": org_item["org_id"], "label": org_label})

        f_chem_list = []

        chem_list = list(org_item["chemicals"].keys())

        # Browse chemical and add them to a list. The list contains tuple (chemical label, len). Indeed, for numbering if there is a contraction, we need to know the number of elements.
        for chem in chem_list:

            chem_label = ""
            # If it is a simple chemical, or, a chemical class:
            if org_item["chemicals"][chem]["suffixes"][0] is None:

                soft_chem_label = chem
                chem_type = org_item["chemicals"][chem]["type"]
                chem_id = org_item["chemicals"][chem]["id"]

                # The prompt_label is used later by the selected to check that the chemical have been mentioned in the generated as expected.
                prompt_label = None

                if chem_type == "chemical":
                    # The orginal chemical label which should be use as ground-truch labels
                    if numbering:
                        chem_label = soft_chem_label + ' ' + "(%d)" %(org_item["chemicals"][chem]["suffixes"][1])
                    else:
                        chem_label = soft_chem_label

                    prompt_label = chem_label

                else:
                    n = org_item["chemicals"][chem]["n"]
                    
                    if numbering:                    
                        prompt_label = soft_chem_label + ' ' + "(%d-%d)" %(org_item["chemicals"][chem]["suffixes"][1], (org_item["chemicals"][chem]["suffixes"][1] + n -1))
                    else:
                        prompt_label = soft_chem_label

                    chem_label = num2words(n) + ' ' + prompt_label

                f_chem_list.append(chem_label)
                all_relations.append([org_id, chem_id])

                # add to all chemicals mentions. Checking that this dict does not already exists in the list
                if soft_chem_label not in all_chemicals_mentions:
                    all_chemicals_mentions[soft_chem_label] = ({"id": org_item["chemicals"][chem]["id"], "label": soft_chem_label, "prompt_label": prompt_label, "type": chem_type})

                continue

            # If there is a possibily of contraction, do it with a proba p_contract. the initial dict need to be kept intact
            indexes_for_contraction = []
            indexes_wo_contraction = []

            for i in range(len(org_item["chemicals"][chem]["suffixes"])):
                # If for a particular nomenclature (roman, ALPHA, num, etc.) there are more than 2 suffixes, then it can be contracted
                if len(org_item["chemicals"][chem]["suffixes"][i]) > 2:
                    indexes_for_contraction.append(i)
                else:
                    indexes_wo_contraction.append(i)
            
            # Apply contraction with a proba p_contract
            if len(indexes_for_contraction) > 0:
                if random.uniform(0,1) <= p_contract:
                    logger.debug("Contraction of consecutive list: True")

                    form = np.random.choice(a=[f[0] for f in CONTRACT_FORM], size=1, p=[f[1] for f in CONTRACT_FORM])[0]

                    for i in indexes_for_contraction:

                        soft_chem_label = chem + 's' + ' ' + form.format(start=org_item["chemicals"][chem]["suffixes"][i][0][0], end=org_item["chemicals"][chem]["suffixes"][i][-1][0])

                        if numbering:
                            chem_label = soft_chem_label + ' ' + "(%d-%d)" %(org_item["chemicals"][chem]["suffixes"][i][0][2], org_item["chemicals"][chem]["suffixes"][i][-1][2])
                        else:
                            chem_label = soft_chem_label
                        
                        prompt_label = chem_label
                        
                        f_chem_list.append(chem_label)

                        # Should the label be the contracted label (e.g flavolones A-F) or the uncompressed list of labels
                        if compressed_labels:

                            # Create an id from the compressed list of chemicals
                            chem_id = org_item["chemicals"][chem]["suffixes"][i][0][1] + "-" + org_item["chemicals"][chem]["suffixes"][i][-1][1]
                            
                            # If it has not already been added for another orgamism
                            if soft_chem_label not in all_chemicals_mentions:
                                all_chemicals_mentions[soft_chem_label] = ({"id": chem_id, "label": soft_chem_label, "prompt_label": prompt_label, "type": "chemical-list"})

                            all_relations.append([org_id, chem_id])

                        # Else, all the uncompressed list is added in the corresponding order, e.g: flavolone A, flavolone B, etc.
                        else:
                            for chem_suffix in org_item["chemicals"][chem]["suffixes"][i]:

                                chem_id = chem_suffix[1]
                                _soft_chem_label = original_chemical_labels[original_chemical_labels["id"] == chem_id]["label"].values[0]

                                all_relations.append([org_id, chem_id])

                                if _soft_chem_label not in all_chemicals_mentions:
                                    all_chemicals_mentions[_soft_chem_label] = ({"id": chem_id, "label": _soft_chem_label, "prompt_label": prompt_label, "type": "chemical"})

                # If no contraction, simply enumerate
                else:
                    logger.debug("Contraction of consecutive list: False")

                    for i in indexes_for_contraction:
                        for s in org_item["chemicals"][chem]["suffixes"][i]:

                            chem_id = s[1]
                            _soft_chem_label = original_chemical_labels[original_chemical_labels["id"] == chem_id]["label"].values[0]

                            if numbering:
                                chem_label = _soft_chem_label + ' ' + "(%d)" %(s[2])
                            else:
                                chem_label = _soft_chem_label
                            
                            prompt_label = chem_label
                            
                            f_chem_list.append(chem_label)

                            all_relations.append([org_id, chem_id])

                            if _soft_chem_label not in all_chemicals_mentions:
                                all_chemicals_mentions[_soft_chem_label] = ({"id": chem_id, "label": _soft_chem_label, "prompt_label": prompt_label, "type": "chemical"})

            # Stringify the remainings (those without contraction anyway)
            for i in indexes_wo_contraction:
                for s in org_item["chemicals"][chem]["suffixes"][i]:

                    chem_id = s[1]
                    _soft_chem_label = original_chemical_labels[original_chemical_labels["id"] == chem_id]["label"].values[0]

                    if numbering:
                        chem_label = _soft_chem_label + ' ' + "(%d)" %(s[2])
                    else:
                        chem_label = _soft_chem_label
                    
                    prompt_label = chem_label

                    f_chem_list.append(chem_label)
                    all_relations.append([org_id, chem_id])

                    if _soft_chem_label not in all_chemicals_mentions:
                        all_chemicals_mentions[_soft_chem_label] = ({"id": chem_id, "label": _soft_chem_label, "prompt_label": prompt_label, "type": "chemical"})
        
        # Drawn a mention:
        mention = np.random.choice(a=[f[0] for f in MENTION_FORM], size=1, p=[f[1] for f in MENTION_FORM])[0]
        
        # handling a singular chem
        if "isolated" in mention and len(f_chem_list) == 1:
            mention = "{chems} was isolated from {org}"
        
        mention_list.append(mention.format(org=org_label, chems=', '.join(f_chem_list[:-1]) + ' and ' + f_chem_list[-1] if len(f_chem_list) > 1 else f_chem_list[0]))
        
        # Put an upper case at first letter:
        for i in range(len(mention_list)):
            mention_list[i] = mention_list[i][0].upper() + mention_list[i][1:]

    final_mention_string = '. '.join(mention_list) + '.'

    # Convert list of all mentions to a set:
    all_chemicals_mentions = list(all_chemicals_mentions.values())

    
    return final_mention_string, all_relations, all_organisms_mentions, all_chemicals_mentions


