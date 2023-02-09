"""Olinda CLI entrypoint."""

import csv
from itertools import chain
from pathlib import Path
import os
import sys

import click
import joblib
from joblib import cpu_count

from compound_embedding.cli.generate import gen
from compound_embedding.cli.test import test
from compound_embedding.cli.train import train
from compound_embedding.pipelines.common import get_all_paths, parallel_on_generic
from compound_embedding.pipelines.data_qc import (
    check_mol_counts,
    fix_corrupted_files,
    remove_part_files,
)

from .. import __version__


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
@click.option("--fix", is_flag=True, help="Fix corrupted files")
@click.option("--jobs", type=int, help="Number of threads to use", default=cpu_count())
def qc(inp: str, out: str, seq: bool, fix: bool, jobs: int) -> None:
    """Run data quality check."""
    saved_corrupted_files_array = Path().joinpath("corrupted_files.joblib")
    if saved_corrupted_files_array.is_file():
        corrupted_file_paths = joblib.load(saved_corrupted_files_array)
    else:
        file_paths = get_all_paths(Path(inp))
        print(f"Number of input files found: {len(file_paths)}")
        if seq:
            corrupted_file_paths, corrupted_file_reasons = check_mol_counts(file_paths, Path(out))
        else:
            worker_out = parallel_on_generic(
                file_paths, check_mol_counts, [Path(out)], cpu_count()
            )
            corrupted_file_paths = [tup[0] for tup in worker_out]
            corrupted_file_reasons = [tup[1] for tup in worker_out]
            corrupted_file_paths = list(chain.from_iterable(corrupted_file_paths))
            corrupted_file_reasons = list(chain.from_iterable(corrupted_file_reasons))
            joblib.dump(corrupted_file_paths, Path().joinpath("corrupted_files.joblib"))
            joblib.dump(corrupted_file_reasons, Path().joinpath("corrupted_file_reasons.joblib"))
    
    if fix:
        print(f"Found {len(corrupted_file_paths)} to fix.")
        # delete part files
        remove_part_files(Path(out))

        # fix corrupted files by re running pipeline for them
        if seq:
            fix_corrupted_files(corrupted_file_paths, Path(inp), ["grover", "mordred"])
        else:
            parallel_on_generic(
                corrupted_file_paths,
                fix_corrupted_files,
                [Path(inp), ["grover", "mordred"]],
                jobs,
            )
            # Delete dumped corrupted files array
            if saved_corrupted_files_array.is_file():
                saved_corrupted_files_array.unlink()
            
            # Return early - do not save corrupted files list as csv
            return

    # Write corrupted files to disk
    wtr = csv.writer(
        open("corrupted_files.csv", "w"), delimiter=",", lineterminator="\n"
    )
    for x in corrupted_file_paths:
        wtr.writerow([x])


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Crux console."""
    pass


main.add_command(gen)
main.add_command(train)
main.add_command(test)
main.add_command(qc)
