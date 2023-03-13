"""Ersilia embeddings cli."""

import csv
import logging
from pathlib import Path
from typing import List, Optional

import click
import numpy as np

from . import __version__
from eosce.models import ErsiliaCompoundEmbeddings


@click.command()
@click.option("--debug", is_flag=True, help="Ouput debug logs.")
@click.option("--grid", is_flag=True, help="Convert embeddings to a grid.")
@click.option("-i", "--inp", type=str, help="Path to CSV input file. The CSV must contain a single colum of SMILES without header")
@click.option(
    "-o",
    "--out",
    type=str,
    help="File path to save results. Extension of the file is used to select the output format.",
)
@click.argument("SMILES", nargs=-1)
def embed(
    debug: bool, grid: bool, inp: Optional[str], out: Optional[str], smiles: List[str]
) -> None:
    """Generate embeddings."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    model = ErsiliaCompoundEmbeddings()

    # Parse input csv if given
    if inp:
        csv_path = Path(inp)
        with csv_path.open() as f:
            file = csv.reader(f)
            smiles = [line[0] for line in file]
    embeddings = model.transform(smiles, grid)

    # Parse output format requested
    if out:
        ext = out.split(".")[-1]
        if ext == "csv":
            with open(out, "w", newline="") as f:
                writer = csv.writer(f)
                for i, row in enumerate(embeddings):
                    writer.writerow([smiles[i], *row])
        elif ext == "npz":
            np.savez_compressed(out, embeddings=embeddings)
        else:
            raise Exception(f"File format {ext} is not supported for saving.")
    else:
        np.set_printoptions(threshold=np.inf)
        print(embeddings)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Ersilia embeddings CLI."""
    pass


main.add_command(embed)
