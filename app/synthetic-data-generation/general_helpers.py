# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import sys
import logging
import json
import random
from collections import defaultdict

def get_logger(name, **kwargs):
    """Create a default logger

    Returns:
        logging.logger: a default logger
    """

    # set loggers
    level = kwargs.get("level", logging.DEBUG)
    log_path = kwargs.get("path", "./prompting-generate-abstract.log")
    open(log_path, 'w', encoding="utf-8").close()

    handlers = [logging.FileHandler(filename=log_path), logging.StreamHandler(stream=sys.stdout)]
    logging.basicConfig(handlers=handlers, format='%(asctime)s %(levelname)-8s %(message)s', level=level)
    logger = logging.getLogger(name)

    return logger

def split_json(input_json_path, out_dir, l=100):
    """Split a json dataset in several batches

    Args:
        input_json_path (str): path to the original dataset json file.
        out_dir (dir): path to the output directory 
        l (int, optional): int. Defaults to 100. number of items in a batch
    """

    f_input_name = os.path.splitext(os.path.basename(input_json_path))[0]

    with open(input_json_path, "r") as f:
        data = json.load(f)
    import random
    list_of_keys = [k for k in data.keys()]
    L = len(list_of_keys)
    n_groups = (L // l) + (L % l > 0)

    for i in range(n_groups):
        sub_list_of_keys = list_of_keys[(i*l):((i+1)*l)]
        sub_dict = {}

        for k in sub_list_of_keys:
            sub_dict[k] = data.pop(k)

        with open(os.path.join(out_dir, f_input_name + "_" + str(i + 1) + ".json"), "w", encoding="utf-8") as f_out:
            json.dump(sub_dict, f_out, indent=4)

def merge_json(input_json_path, out_file_name):
    """Merge dataset batches in one dataset.

    Args:
        input_json_path (str): path to the directory containing all json batches
        out_file_name (_type_):path to the output file name to write the merged dataset 
    """
    
    # browse and merge
    g = dict()

    for sub_json_f in os.listdir(input_json_path):

        if not sub_json_f.endswith(".json"):
            continue

        print("Add " + sub_json_f)
        with open(os.path.join(input_json_path, sub_json_f), "r", encoding="utf-8") as sub_json:
            g.update(json.load(sub_json))

    with open(out_file_name, "w", encoding="utf-8") as f_out:
            json.dump(g, f_out, indent=4)


def create_cache(inputed_json_path, outputed_json_path, out_dir):
    with open(inputed_json_path, "r", encoding="utf-8") as f1:
        inputed_json = json.load(f1)
    
    all_pmids = list(inputed_json.keys())
    
    with open(outputed_json_path, "r", encoding="utf-8") as f1:
        outputed_json = json.load(f1)
    
    cache = {}

    for pmid in all_pmids:
        list_of_abstract = []
        for abtract_item in outputed_json.values():
            if abtract_item["PMID"] == pmid:
                list_of_abstract.append({"AbstractText": abtract_item["AbstractText"], "temperatue": abtract_item["temperatue"]})
        cache[pmid] = list_of_abstract
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    with open(os.path.join(out_dir, "abstracts.json"), "w", encoding="utf-8") as f_out:
            json.dump(cache, f_out, indent=4)



# RUN: merge_dataset_into_large(list_of_dataset_path, out_dir, split=[0.95, 0.05])

def merge_dataset_into_large(list_of_dataset_path, out_dir, split = [0.9, 0.1]):

    g = defaultdict(lambda: list())

    for dataset_path in list_of_dataset_path:
        sub_g = dict()
        
        print("read " + dataset_path)
        with open(dataset_path, "r", encoding="utf-8") as sub_json:
            sub_g = json.load(sub_json)
        
        for k, v in sub_g.items():
            pmid, id = k.split('-')
            g[pmid].append(v)
    
    all_pmids = list(g.keys())
    print("Total len:" + str(len(all_pmids)))

    assert sum(split) == 1
    train_len = int(split[0] * len(all_pmids))
    valid_len = len(all_pmids) - train_len

    print("Train set size: " + str(train_len))
    print("Valid set size: " + str(valid_len))

    valid_index = random.sample(range(len(all_pmids)), valid_len)
    valid_set = dict()
    COUNT_VALID = 0
    for v_i in valid_index:
        valid_pmid = all_pmids[v_i]
        valid_items = g.pop(valid_pmid)
        for i in range(len(valid_items)):
            valid_set[valid_pmid + '-' + str(i)] = valid_items[i]
        COUNT_VALID += len(valid_items)
    
    COUNT_TRAIN = 0
    train_set = dict()
    for pmid, train_items in g.items():
        for j in range(len(train_items)):
            train_set[pmid + '-' + str(j)] = train_items[j]
        COUNT_TRAIN += len(train_items)
    
    print("Number of training elements: " + str(COUNT_TRAIN))
    print("Number of validation elements: " + str(COUNT_VALID))
    # Export
    with open(os.path.join(out_dir, "train.json"), "w", encoding="utf-8") as f_train_out:
        json.dump(train_set, f_train_out, indent=4)

    with open(os.path.join(out_dir, "valid.json"), "w", encoding="utf-8") as f_valid_out:
        json.dump(valid_set, f_valid_out, indent=4)
    
    return True

