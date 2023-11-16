# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import os
import argparse
import json

from utils import parse_predictions, parse_reference

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="path to the input directory containing results json files", type=str, required=True, dest='input')
parser.add_argument("--test-file", help="path to the original test file with ground-truth labels", type=str, required=True, dest='test_file')
parser.add_argument("--output-file", help="path to the output directoy", type=str, required=True, dest='output_file')
args = parser.parse_args()

input_file = args.input

output_dir = os.path.dirname(args.output_file)

test_file = args.test_file

# Read input data
with open(test_file, "r", encoding="utf-8") as f_test:
    test_ref = json.load(f_test)

with open(input_file, "r", encoding="utf-8") as f_in:
    g_dict = json.load(f_in)

INFERENCE_FAILED_LIST = []
PARSING_FAILED_LIST = []
PARSING_SUCCEED_LIST = []

total_true_positives = 0
total_false_positives = 0
total_positives = 0
micro_f1 = 0
micro_precision = 0
micro_recall = 0

for pmid, item in g_dict.items():

    if pmid not in test_ref:
        print("[WARNING] PMID " + pmid + " has no reference in the reference test set.")
        continue

    # Parse ref
    ref = test_ref[pmid]
    parsed_reference = parse_reference(ref)

    # Increment number of positives
    total_positives += sum([len(l) for l in parsed_reference.values()])

    # Did it failed during inference ?
    if not "choices" in item:
        print("[WARNING] PMID " + pmid + " failed during inference. No available results.")
        INFERENCE_FAILED_LIST.append(pmid)
        continue
    
    # If it didn't failed, try to parse
    parsed_predictions = parse_predictions(item["choices"][0]["text"], ". ", "produces")

    if not len(parsed_predictions):
        print("[WARNING] PMID " + pmid + " added to parsing failed list.")
        PARSING_FAILED_LIST.append(pmid)
        continue

    # predictions are not empty, evaluate against reference:
    PARSING_SUCCEED_LIST.append(pmid)
    
    FP = 0
    TP = 0

    for pred_org, pred_chem_list in parsed_predictions.items():
        for pred_chem in pred_chem_list:

            # Check if it is a TP
            if pred_chem in parsed_reference[pred_org]:
                TP += 1
            else:
                print(pred_chem)
                FP += 1

    total_true_positives += TP
    total_false_positives += FP

if total_true_positives != 0 or total_false_positives != 0:
    micro_precision = total_true_positives / (total_true_positives + total_false_positives)

if total_positives != 0:
    micro_recall = total_true_positives / total_positives

if micro_precision != 0 or micro_recall != 0:
    micro_f1 = 2 * ((micro_precision * micro_recall) / (micro_precision + micro_recall))


out = {"micro-precision": round(micro_precision * 100, 2), "micro-recall": round(micro_recall * 100, 2), "micro-f1": round(micro_f1 * 100, 2), "parsing-ok": len(PARSING_SUCCEED_LIST), "parsing-failed": len(PARSING_FAILED_LIST)}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(args.output_file, "w", encoding="utf-8") as f_out:
    json.dump(out, f_out, indent=4)