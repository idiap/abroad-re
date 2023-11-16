# SPDX-FileCopyrightText: 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Delmas Maxime maxime.delmas@idiap.ch
#
# SPDX-License-Identifier: GPL-3.0-or-later

import argparse 
import optuna
import os
import argparse
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback

# Get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n-trials", help="Number of trial for the study", type=int, required=True, dest='n_trials')
parser.add_argument("--output-serialize", help="path to the output dir for serialization (and to be remove after)", type=str, required=True, dest='output_ser')
parser.add_argument("--config", help="path to the config file", type=str, required=True, dest='config_file')
parser.add_argument("--out-dir", help="path to the output dir", type=str, required=True, dest='output_dir')

args = parser.parse_args()

output_dir = args.output_dir

output_ser = args.output_ser

n_trials = args.n_trials

MAX_CALL_BACK = 30

def objective(trial: optuna.Trial) -> float:
    trial.suggest_float("lr_decoder", 1e-6, 1e-3, log=True)
    trial.suggest_categorical("beam_size", [3, 5])
    trial.suggest_float("length_penality", 1, 3, step=0.5)

    executor = optuna.integration.allennlp.AllenNLPExecutor(
        trial=trial,  # trial object
        config_file=args.config_file,  # jsonnet path
        serialization_dir=os.path.join(output_ser, str(trial.number)),  # directory for snapshots and logs
        metrics="best_validation_fscore",
        include_package="seq2rel"
    )

    return executor.run()


study_name = "seq2rel-hp-ft"
storage_name = f"sqlite:///{os.path.join(args.output_dir, study_name)}.db"

study = optuna.create_study(study_name=study_name, 
    storage=storage_name, 
    sampler=TPESampler(),
    direction="maximize", 
    load_if_exists=True)

study.optimize(objective, 
    n_trials=n_trials, 
    gc_after_trial=True,
    callbacks=[MaxTrialsCallback(MAX_CALL_BACK, states=None)])