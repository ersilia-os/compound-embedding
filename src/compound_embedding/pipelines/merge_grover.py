"""Merge Grover descriptors with FS-Mol dataset."""

from pathlib import Path
from typing import List
import os

from groverfeat import Featurizer

from compound_embedding.pipelines.common import read_as_jsonl, write_jsonl_gz_data


def gen_grover_merged_files(task_file_paths: List[Path], output_dir: Path, rm_input: bool = False) -> None:
    """Merge fs-mol with grover fingerprints.

    Args:
        task_file_paths (List[Path]): List of all tasks.
        output_dir (Path): Path to save generated files
    """
    # Ensure dirs are present
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.joinpath("train"), exist_ok=True)
    os.makedirs(output_dir.joinpath("test"), exist_ok=True)
    os.makedirs(output_dir.joinpath("valid"), exist_ok=True)
    grover_model = Featurizer()
    for path in task_file_paths:
        output_file_path = output_dir.joinpath(*path.parts[-2:])
        output_file_path_temp = output_file_path.with_suffix(".part")
        if output_file_path.is_file():
            print(f"Path: {output_file_path} | Already processed")
            continue
        samples = []
        sample_smiles = []
        modified_data = []
        output_file_path = output_dir.joinpath(*path.parts[-2:])
        for sample in read_as_jsonl(path):
            samples.append(sample)
            sample_smiles.append(sample["SMILES"])

        # Generate grover fingerprints in batch
        grover_fps = grover_model.transform(sample_smiles)

        # Update samples with grover data
        for i, sample in enumerate(samples):
            sample["grover"] = grover_fps[i]
            modified_data.append(sample)

        # Write modified files
        write_jsonl_gz_data(output_file_path_temp, modified_data, len(modified_data))
        output_file_path_temp.rename(output_file_path)

        # If remove input is True then delete the input files
        if rm_input:
            os.remove(path)