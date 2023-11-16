# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import pandas as pd

from collections import defaultdict

def parse_reference(ref):

    def white_space_fix(text):
        return "".join(text.split())

    parsed_ref = defaultdict(lambda: [])

    chemicals_df = pd.DataFrame(ref["chemicals"]).fillna("")
    organisms_df = pd.DataFrame(ref["organisms"]).fillna("")

    for r in ref["relations"]:
        org_label = white_space_fix(organisms_df[organisms_df["id"] == r[0]]["label"].values[0].lower())
        chem_label = white_space_fix(chemicals_df[chemicals_df["id"] == r[1]]["label"].values[0].lower())
        parsed_ref[org_label].append(chem_label)

    return parsed_ref

def parse_predictions(prediction_str, separator, template_sep):
    
    def remove_articles(text):
        regex = re.compile(r"\b(an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return "".join(text.split())

    def remove_unexpected_formating(text):
        # For the potential 1. ... 2. ....
        text = re.sub("\d\.\s", "", text)
        text = re.sub("\*\s", "", text)
        text = re.sub("\d\)\s", "", text)
        return text

    relations = defaultdict(lambda: [])
    prediction_str = remove_unexpected_formating(prediction_str)
    items = prediction_str.strip().split(separator)

    if not len(items):
        print(f"Output predictions '%s', cannot be parsed with separator %s" %(prediction_str, separator))
        return relations

    for item in items:

        # Template 1 (the expected one): ' ... ORG produces CHEM .. ' 
        try:
            org, chem = item.split("produces")

        except ValueError as e_fail_template_1:

            print("Fail during parsing of output prediction: %s.\n%s" %(item, str(e_fail_template_1)))

            # Template 2 (unexpected): ' ... ORG produces CHEM .. '
            try:
                chem, org  = item.split("is produced by")

            except ValueError as e_fail_template_2:
                print("Fail during parsing of output prediction: %s.\n%s" %(item, str(e_fail_template_2)))
                
                try:
                    chem, org  = item.split("are produced by")

                except ValueError as e_fail_template_3:
                    print("Fail during parsing of output prediction: %s.\n%s" %(item, str(e_fail_template_3)))
                    continue

        org = white_space_fix(remove_articles(org.strip('\n .').lower()))
        chem = white_space_fix(remove_articles(chem.strip('\n .').lower()))

        if chem not in relations[org]:
            relations[org].append(chem)

    return relations