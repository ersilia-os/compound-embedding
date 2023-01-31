"""Data quality checks."""

import os
from pathlib import Path
from typing import List, Literal

from tqdm import tqdm

from compound_embedding.pipelines.common import read_as_jsonl


def check_mol_counts(in_files: List[Path], out_path: Path) -> List[Path]:
    print(f"Number of input files found: {len(in_files)}")
    corrupted_files = []
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
                
                if in_samples != out_samples:
                    corrupted_files.append("/".join(path.parts[-2:]))
                    print(f"Found corrupted sample: {'/'.join(path.parts[-2:])}")
            except Exception as e:
                print(e)
                corrupted_files.append("/".join(path.parts[-2:]))
                print(f"Found corrupted sample: {'/'.join(path.parts[-2:])}")
                
        else:
            corrupted_files.append("/".join(path.parts[-2:]))
            print(f"Found corrupted sample: {'/'.join(path.parts[-2:])}")
            
    return corrupted_files


def remove_part_files(in_dir: List[Path]) -> None:
    """Remove partly generated files.

    Args:
        in_dir (List[Path]): Input directory to remove files from
    """
    path_gen = Path(in_dir).glob('**/*.part')
    files = [x for x in path_gen if x.is_file()]
    for file in files:
        os.remove(file)


def fix_corrupted_files(corrupted_files: List[Path], input_dir: Path, output_dir: Path, pipelines: List[Literal["grover", "mordred"]]) -> None:
    """Fix corrupted files.

    Args:
        corrupted_files (List[Path]): Corrupted file paths.
        input_dir (Path): Directory containing source files.
        output_dir (Path): Directory containing corrupted files
        peipelines (List[Literal["grover", "mordred"]]): Pipelines to run to fix corrupted files
    """
    pass