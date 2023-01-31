"""Olinda CLI entrypoint."""

import csv
from itertools import chain
from pathlib import Path

import click
from joblib import cpu_count

from compound_embedding.cli.generate import gen
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
    file_paths = get_all_paths(Path(inp))
    print(f"Number of input files found: {len(file_paths)}")
    if seq:
        corrupted_file_paths = check_mol_counts(Path(inp), Path(out))
    else:
        corrupted_file_paths = parallel_on_generic(
            file_paths, check_mol_counts, [Path(out)], jobs
        )
        corrupted_file_paths = list(chain.from_iterable(corrupted_file_paths))

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
                [Path(out), ["grover", "mordred"]],
                jobs,
            )
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
main.add_command(qc)
