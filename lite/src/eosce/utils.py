"""Utils."""

from pathlib import Path
from typing import List

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator


def get_package_root_path() -> Path:
    """Get path of the package root.

    Returns:
        Path: Package root path
    """
    return Path(__file__).parent.absolute()


def smiles_to_morgan(smiles: List[str]) -> List[np.ndarray]:
    """Convert smiles to Morgan fingerprints."""
    fingerprints = []
    for smile in smiles:
        fingerprint_vect = rdFingerprintGenerator.GetCountFPs(
            [MolFromSmiles(smile)], fpType=rdFingerprintGenerator.MorganFP
        )[0]
        fingerprint = np.zeros((0,), np.int8)  # Generate target pointer to fill
        DataStructs.ConvertToNumpyArray(fingerprint_vect, fingerprint)
        fingerprints.append(fingerprint)
    return np.array(fingerprints)


#  Adapted from https://github.com/ersilia-os/griddify
class Flat2Grid(object):
    def __init__(self, mappings, side):
        self._mappings = mappings
        self._side = side

    def transform(self, X):
        X = np.array(X)
        Xt_sum = np.zeros((X.shape[0], self._side, self._side))
        Xt_cnt = np.zeros(Xt_sum.shape, dtype=int)
        for i in range(X.shape[0]):
            x = X[i, :]
            for j, v in enumerate(x):
                idx_i, idx_j = self._mappings[j]
                Xt_sum[i, idx_i, idx_j] += v
                Xt_cnt[i, idx_i, idx_j] += 1
        Xt = Xt_sum / Xt_cnt
        return Xt
