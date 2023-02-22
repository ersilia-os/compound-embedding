"""Generate training data pipeline."""

from pathlib import Path
from typing import Dict, List, Optional

from groverfeat import Featurizer
import joblib
import numpy as np
import torch
from tqdm import tqdm

from compound_embedding.fs_mol.utils.protonet_utils import PrototypicalNetworkTrainer
from compound_embedding.fs_mol.utils.torch_utils import torchify
from compound_embedding.pipelines.common import get_package_root_path
from compound_embedding.utils import smiles_to_morgan, read_as_h5, write_h5_data


def gen_training_data(
    smiles_list: List[str],
    output_file: Path,
    chunk_size: int = 1000,
    parallel: bool = False,
    global_index_map: Optional[Dict[str, int]] = None,
    dataset_len: Optional[int] = None,
) -> None:
    """Generate training dataset.

    Args:
        smiles_list (List[Path]): List of all tasks.
        output_file (Path): Path to save generated file.
        chunk_size (int): Size of chunks to calculate at a time.
        parallel (bool): Use parallel writers. Defaults to False.
        global_index_map (Optional[Dict[str, int]]): Optional global index map to use for parellel writing.
            Deafaults to None.
        dataset_len (Optional[int]): Length of the entire dataset for parallel writing.
    """
    if parallel:
        assert global_index_map is not None
        assert dataset_len is not None

    else:
        dataset_len = len(smiles_list)

    grover_data = []
    mordred_data = []
    grover_calc = Featurizer()
    mordred_calc = joblib.load(get_package_root_path() / "mordred_descriptor.joblib")
    protonet_model = PrototypicalNetworkTrainer.build_from_model_file(
        (get_package_root_path() / "fully_trained.pt"), device="cuda"
    )
    protonet_model.eval()
    protonet_model.to("cuda")
    chunks = []
    for i in range(0, len(smiles_list), chunk_size):
        chunks.append(slice(i, i + chunk_size))

    field_names = ["morgan", "grover", "mordred", "protonet", "check"]
    field_shapes = [(2048,), (5000,), (1379,), (1024,), (1,)]
    field_types = [np.int8, np.float32, np.float32, np.float32, np.bool8]
    for chunk in tqdm(chunks):
        start_smile = smiles_list[chunk][0]
        stop_smile = smiles_list[chunk][-1]
        if is_chunk_processed(
            [global_index_map[start_smile], global_index_map[stop_smile]],
            output_file,
            ["check"],
        ):
            print(
                f"Chunk: {chunk} with global [{global_index_map[start_smile]}, {global_index_map[stop_smile]}] is Already processed!"
            )
        else:
            print(
                f"Processing Chunk: {chunk} with global [{global_index_map[start_smile]}, {global_index_map[stop_smile]}]"
            )
            grover_data = grover_calc.transform(smiles_list[chunk])
            mordred_data = mordred_calc.transform(smiles_list[chunk])
            morgan_data = smiles_to_morgan(smiles_list[chunk])
            with torch.no_grad():
                protonet_data = protonet_model.fc(
                    torch.cat(
                        [
                            torchify(grover_data, "cuda"),
                            torchify(morgan_data, "cuda"),
                            torchify(mordred_data.to_numpy(), "cuda"),
                        ],
                        dim=1,
                    )
                )
            protonet_data = protonet_data.to("cpu").numpy()
            data = [
                {
                    "morgan": morgan_data[i],
                    "grover": grover_data[i],
                    "mordred": mordred_data.iloc[
                        i,
                    ].to_numpy(),
                    "protonet": protonet_data[i],
                    "check": 1,
                    "ele_index": global_index_map[smiles_list[chunk][i]],
                }
                for i in range(len(smiles_list[chunk]))
            ]
            write_h5_data(
                output_file,
                data,
                field_names,
                field_shapes,
                field_types,
                parallel,
                index_field="ele_index",
                dataset_len=dataset_len,
            )


def is_chunk_processed(
    read_indices: List[int], filepath: Path, fieldnames: List[str]
) -> bool:
    """Check if chunk is already processed.

    Args:
        read_indices (List[int]): List of indixes to check.
        filepath (Path): File to read.
        fieldnames (List[str]): Fieldnames to load datasets.

    Returns:
        bool: True if chunk is already processed.
    """
    if filepath.is_file():
        read_iter = read_as_h5(filepath, fieldnames, read_indices=read_indices)
        check_array = []
        for ele in tqdm(read_iter, desc="Checking datapoints"):
            check_array.append(ele["check"][0])
        if sum(check_array) == len(read_indices):
            return True
        else:
            return False
    else:
        return False
