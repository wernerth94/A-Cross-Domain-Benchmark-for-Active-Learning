import argparse
import sys, shutil
import os
from os.path import join, exists
import pandas as pd

def check_for_nan_cols(df:pd.DataFrame):
    cleaned_pd = df.copy(deep=True)
    for col_name in df:
        if df[col_name].isnull().values.any():
            cleaned_pd.drop(columns=[col_name], inplace=True)
    return cleaned_pd


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DNA")
args = parser.parse_args()

dataset_folder = join("runs", args.dataset)
assert exists(dataset_folder)

output_folder = join(dataset_folder, "Oracle")
if exists(output_folder):
    resp = input(f"{output_folder} already exists. Do you want to remove it? (y/n)")
    if resp != "y":
        sys.exit(0)
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

result_acc = pd.DataFrame()
result_loss = pd.DataFrame()

for run_folder in os.listdir(dataset_folder):
    if run_folder.startswith("Oracle_"):
        accuracies = pd.read_csv(join(dataset_folder, run_folder, "accuracies.csv"), header=0, index_col=0)
        accuracies = check_for_nan_cols(accuracies)
        result_acc = pd.concat([result_acc, accuracies], axis=1, ignore_index=True)
        losses = pd.read_csv(join(dataset_folder, run_folder, "losses.csv"), header=0, index_col=0)
        losses = check_for_nan_cols(losses)
        result_loss = pd.concat([result_loss, losses], axis=1, ignore_index=True)

result_acc.to_csv(join(output_folder, "accuracies.csv"))
result_loss.to_csv(join(output_folder, "losses.csv"))

