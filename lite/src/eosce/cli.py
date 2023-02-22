"""Ersilia embeddings cli."""

import logging
from typing import List, Optional

import click
import numpy as np

from . import __version__
from eosce.models import ErsiliaCompoundEmbeddings

@click.command()
@click.option("--debug", is_flag=True, help="Ouput debug logs.")
@click.option("--out", type=str, help="File path to save results.")
@click.argument("SMILES", nargs=-1)
def embed(debug: bool, out: Optional[str], smiles: List[str]) -> None:
    """Generate embeddings."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    model = ErsiliaCompoundEmbeddings()
    embeddings = model.transform(smiles)
    if out:
        np.savez_compressed(out, embeddings=embeddings)
    else:
        np.set_printoptions(threshold=np.inf)
        print(embeddings)

@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Ersilia embeddings CLI."""
    pass

main.add_command(embed)