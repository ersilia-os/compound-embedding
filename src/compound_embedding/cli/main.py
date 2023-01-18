"""Olinda CLI entrypoint."""

import click

from compound_embedding.cli.generate import gen

from .. import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Crux console."""
    pass


main.add_command(gen)
