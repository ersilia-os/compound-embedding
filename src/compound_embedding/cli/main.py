"""Olinda CLI entrypoint."""

import csv
from itertools import chain
from pathlib import Path

import click

from compound_embedding.cli.generate import gen
from compound_embedding.cli.train import train
from compound_embedding.pipelines.common import parallel_on_generic
from compound_embedding.pipelines.data_qc import check_mol_counts, get_all_paths

from .. import __version__

@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
def qc(inp: str, out: str, seq: bool) -> None:
    """Run data quality check."""
    file_paths = get_all_paths(Path(inp))
    print(f"Number of input files found: {len(file_paths)}")
    if seq:
        corrupted_file_paths = check_mol_counts(Path(inp), Path(out))
    else:
        corrupted_file_paths = parallel_on_generic(file_paths, check_mol_counts, [Path(out)], 16)
        corrupted_file_paths = list(chain.from_iterable(corrupted_file_paths))

    wtr = csv.writer(open ('corrupted_files.csv', 'w'), delimiter=',', lineterminator='\n')
    for x in corrupted_file_paths : wtr.writerow([x])


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Crux console."""
    pass


main.add_command(gen)
main.add_command(train)
main.add_command(qc)
