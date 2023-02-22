"""Ersilia embeddings cli."""

import click

from . import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Ersilia embeddings CLI."""
    pass
