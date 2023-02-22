"""Data quality checks."""

import os
from pathlib import Path
from typing import List, Tuple

from groverfeat import Featurizer
import joblib
import pandas as pd
from tqdm import tqdm

from compound_embedding.pipelines.common import (
    get_package_root_path,
    read_as_jsonl,
    write_jsonl_gz_data,
)


def check_mol_counts(
    in_files: List[Path], out_path: Path
) -> Tuple[List[Path], List[str]]:
    print(f"Number of input files found: {len(in_files)}")
    corrupted_files = []
    corrupt_reasons = []
    for path in tqdm(in_files, total=len(in_files)):
        output_file_path = out_path.joinpath(*path.parts[-2:])
        if output_file_path.is_file():
            in_samples = []
            out_samples = []

            # Read input samples
            try:
                for sample in read_as_jsonl(path):
                    in_samples.append(sample["SMILES"])

                # Read output samples
                for sample in read_as_jsonl(output_file_path):
                    out_samples.append(sample["SMILES"])

                if in_samples == out_samples:
                    continue
                else:
                    corrupted_files.append(output_file_path)
                    corrupt_reasons.append("UNMATCHED")

            except Exception:
                # print(e)
                corrupted_files.append(output_file_path)
                corrupt_reasons.append("CORRUPT")

        else:
            corrupted_files.append(output_file_path)
            corrupt_reasons.append("NOT_FOUND")

    return corrupted_files, corrupt_reasons


def remove_part_files(in_dir: List[Path]) -> None:
    """Remove partly generated files.

    Args:
        in_dir (List[Path]): Input directory to remove files from
    """
    path_gen = Path(in_dir).glob("**/*.part")
    files = [x for x in path_gen if x.is_file()]
    for file in files:
        os.remove(file)


def fix_corrupted_files(
    corrupted_files: List[Path],
    input_dir: Path,
    pipelines: List[str],
    rm_input: bool = False,
) -> None:
    """Fix corrupted files.

    Args:
        corrupted_files (List[Path]): Corrupted file paths.
        input_dir (Path): Directory containing source files.
        pipelines (List[str]): Pipelines to run to fix corrupted files.
        rm_input (bool): Remove the source file to save space.
    """
    mordred = joblib.load(get_package_root_path() / "mordred_descriptor.joblib")
    grover_model = Featurizer()

    for path in corrupted_files:
        input_file_path = input_dir.joinpath(*path.parts[-2:])
        output_file_path = path
        output_file_path_temp = output_file_path.with_suffix(".part")
        samples = []
        sample_smiles = []
        # Read samples from the source file
        for sample in read_as_jsonl(input_file_path):
            samples.append(sample)
            sample_smiles.append(sample["SMILES"])

        if "grover" in pipelines:
            modified_data = []
            # Generate grover fingerprints in batch
            grover_fps = grover_model.transform(sample_smiles)

            # Update samples with grover data
            for i, sample in enumerate(samples):
                sample["grover"] = grover_fps[i]
                modified_data.append(sample)

            samples = modified_data

        if "mordred" in pipelines:
            # Generate mordred descriptors in chunked batches
            modified_data = []
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

            samples = modified_data

        # Write modified files
        print(f"Writing file to {output_file_path_temp}")
        write_jsonl_gz_data(output_file_path_temp, samples, len(modified_data))
        print(f"Renaming file to {output_file_path}")
        os.rename(output_file_path_temp, output_file_path)

        # If remove input is True then delete the input files
        if rm_input:
            print(f"Removing original file: {input_file_path}")
            os.remove(input_file_path)
