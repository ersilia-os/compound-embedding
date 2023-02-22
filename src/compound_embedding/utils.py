"""Utility functions."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Optional

from groverfeat import Featurizer
import h5py as h5
import joblib
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
import torch
from tqdm import tqdm

from compound_embedding.fs_mol.utils.torch_utils import torchify
from compound_embedding.pipelines.common import get_package_root_path


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


def smiles_to_proto_input(
    smiles: List[str], device: str = "cuda", m_workers: Optional[int] = None
) -> List[np.ndarray]:
    """Prepare input for protonet from smiles."""
    morgan = smiles_to_morgan(smiles)
    mordred_calc = joblib.load(get_package_root_path() / "mordred_descriptor.joblib")
    mordred = mordred_calc.transform(smiles, m_workers).to_numpy()
    grover_calc = Featurizer()
    grover = grover_calc.transform(smiles)
    return torch.cat(
        [torchify(grover, device), torchify(morgan, device), torchify(mordred, device)],
        dim=1,
    )


def write_h5_data(
    file_path: Path,
    data: Iterable[Dict[str, Any]],
    field_names: List[str],
    field_shapes: List[Tuple],
    field_types: List[np.dtype],
    parallel: bool = False,
    index_field: Optional[str] = None,
    dataset_len: Optional[int] = None,
    error_handling: Optional[Callable[[str, Exception], None]] = None,
) -> None:
    """Write data as hdf5 file.

    Args:
        file_path (Path): Path of the hdf5 file.
        data (Iterable[Dict[str, Any]]): Dataset to save.
        field_names (List[str]): List of key names of the element dictionary to write.
        field_shapes: List[Tuple]: List of field shape for proper data formatting.
        field_types: List[np.dtype]: List of field dtypes for proper data formatting.
        parallel (bool): Use parallel writers. Defaults to False.
        index_field (Optional[str]): Optional index field to use for parellel writing.
            Deafaults to None.
        dataset_len (Optional[int]): Length of the entire dataset for parallel writing.
        error_handling (Optional[Callable[[str, Exception], None]], optional): Function to handle errors. Defaults to None.

    Raises:
        Exception : Write exception.
    """
    file_exist_bool = file_path.is_file()

    if parallel:
        from mpi4py import MPI
        if file_exist_bool:
            h5_file = h5.File(file_path, "r+", driver="mpio", comm=MPI.COMM_WORLD)
        else:
            h5_file = h5.File(file_path, "w", driver="mpio", comm=MPI.COMM_WORLD)
    else:
        if file_exist_bool:
            h5_file = h5.File(file_path, "r+")
        else:
            h5_file = h5.File(file_path, "w")

    if parallel:
        assert dataset_len is not None
    else:
        dataset_len = len(data)

    if not file_exist_bool:
        _ = [
            h5_file.require_dataset(
                fieldname,
                (dataset_len, *shape),
                dtype=dtype,
                chunks=True,
                maxshape=(None, *shape),
            )
            for fieldname, shape, dtype in zip(field_names, field_shapes, field_types)
        ]
    for i, ele in tqdm(enumerate(data), desc="Writing data"):
        try:
            for fieldname in field_names:
                if parallel:
                    dataset = h5_file[fieldname]
                    # with dataset.collective:
                    dataset[ele[index_field], :] = ele[fieldname]
                else:
                    h5_file[fieldname][i, :] = ele[fieldname]
        except Exception as e:
            if error_handling:
                error_handling(e, ele)
            else:
                h5_file.close()
                raise
    h5_file.close()


def read_as_h5(
    file_path: Path,
    dataset_names: List[str],
    read_indices: Optional[List[int]] = None,
    read_index_slice: Optional[slice] = None,
    error_handling: Optional[Callable[[int, Any, Exception], None]] = None,
) -> Iterable[Any]:
    """Read data from hdf5 file.

    Args:
        file_path (Path): Path of the hdf5 file.
        dataset_names (List[str]): List of key names to read as datasets.
        read_indices (Optional[List[int]]): List of indices to read
        read_index_slice (Optional[slice]): Slice of indices to read (Used in parallel workers).
            Defaults to None.
        error_handling (Optional[Callable[[int, Any, Exception], None]], optional): Function to handle errors.
            Defaults to None.

    Raises:
        Exception : Read exception.

    Yields:
        Iterator[Iterable[Any]]: Iterator that returns datapoints.
    """
    h5_file = h5.File(file_path, mode="r")
    if read_indices:
        index_iter = read_indices
    elif read_index_slice:
        index_iter = range(
            read_index_slice.start,
            min(read_index_slice.stop, h5_file[dataset_names[0]].len()),
        )
        print(
            f"Using read index: {read_index_slice.start}, {min(read_index_slice.stop, h5_file[dataset_names[0]].len())}"
        )
    else:
        index_iter = range(h5_file[dataset_names[0]].len())
    try:
        for i in index_iter:
            try:
                # Read values from h5py
                yield {dataset: h5_file[dataset][i] for dataset in dataset_names}
            except Exception as e:
                if error_handling is None:
                    raise
                else:
                    error_handling(i, h5_file, e)
    finally:
        h5_file.close()
