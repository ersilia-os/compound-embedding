"""Merge Mordred descriptors with FS-Mol."""

import os
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

from compound_embedding.pipelines.common import (
    get_package_root_path,
    read_as_jsonl,
    write_jsonl_gz_data,
)


# PROCESSING FUNCTIONS

MAX_NA = 0.2


class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

    def fit(self, X):
        max_na = int((1 - MAX_NA) * X.shape[0])
        idxs = []
        for j in range(X.shape[1]):
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Scaler(object):
    def __init__(self):
        self._name = "scaler"
        self.abs_limit = 10
        self.skip = False

    def set_skip(self):
        self.skip = True

    def fit(self, X):
        if self.skip:
            return
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        if self.skip:
            return X
        X = self.scaler.transform(X)
        X = np.clip(X, -self.abs_limit, self.abs_limit)
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Imputer(object):
    def __init__(self):
        self._name = "imputer"
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class VarianceFilter(object):
    def __init__(self):
        self._name = "variance_filter"

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = self.sel.transform([[i for i in range(X.shape[1])]]).ravel()

    def transform(self, X):
        return self.sel.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


# MORDRED DESCRIPTORS


def mordred_featurizer(smiles):
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas([Chem.MolFromSmiles(smi) for smi in smiles])
    return df


class MordredDescriptor(object):
    def __init__(self):
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.scaler = Scaler()

    def fit(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        return pd.DataFrame(X, columns=self.features)

    def transform(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        X = self.scaler.transform(X)
        return pd.DataFrame(X, columns=self.features)


def gen_mordred_merged_files(
    task_file_paths: List[Path], output_dir: Path, rm_input: bool = False
) -> None:
    """Merge fs-mol with mordred descriptors.

    Args:
        task_file_paths (List[Path]): List of all tasks.
        output_dir (Path): Path to save generated files.
        rm_input (bool): Remove the source file to save space.
    """
    # Ensure dirs are present
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.joinpath("train"), exist_ok=True)
    os.makedirs(output_dir.joinpath("test"), exist_ok=True)
    os.makedirs(output_dir.joinpath("valid"), exist_ok=True)
    mordred = joblib.load(get_package_root_path() / "mordred_descriptor.joblib")
    for path in task_file_paths:
        samples = []
        sample_smiles = []
        modified_data = []
        output_file_path = output_dir.joinpath(*path.parts[-2:])
        output_file_path_temp = output_file_path.with_suffix(".part")
        if output_file_path.is_file():
            print(f"Path: {output_file_path} | Already processed")
            continue
        print(f"Processing: {''.join(path.parts[-1:])}")
        for sample in read_as_jsonl(path):
            samples.append(sample)
            sample_smiles.append(sample["SMILES"])

        # Generate mordred descriptors in chunked batches
        chunks = []
        mordred_df = pd.DataFrame()
        for i in range(0, len(sample_smiles), 1000):
            chunks.append(slice(i, i + 1000))
        for chunk in chunks:
            chunk_df = mordred.transform(sample_smiles[chunk])
            mordred_df = pd.concat([mordred_df, chunk_df])

        # Update samples with mordred data
        for i, sample in enumerate(samples):
            sample["mordred"] = mordred_df.iloc[
                i,
            ].to_numpy()
            modified_data.append(sample)

        # Write modified files
        write_jsonl_gz_data(output_file_path_temp, modified_data, len(modified_data))
        output_file_path_temp.rename(output_file_path)

        # If remove input is True then delete the input files
        if rm_input:
            os.remove(path)
