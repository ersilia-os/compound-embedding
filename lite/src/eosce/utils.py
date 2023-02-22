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