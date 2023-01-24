"""Generate datasets."""

from pathlib import Path

import click

from compound_embedding.pipelines.merge_fs_mol import (
    parallel_on_paths,
    get_all_paths,
    gen_grover_merged_files,
    gen_mordred_merged_files
)


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
def grover(inp: str, out: str, seq: bool) -> None:
    """Generate grover dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_grover_merged_files(file_paths, Path(out))
    else:
        parallel_on_paths(file_paths, gen_grover_merged_files, [Path(out)], 12)


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
def mordred(inp: str, out: str, seq: bool) -> None:
    """Generate mordred dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_mordred_merged_files(file_paths, Path(out))
    else:
        parallel_on_paths(file_paths, gen_mordred_merged_files, [Path(out)], 12)


@click.group()
def gen() -> None:
    """Generate dataset commands."""
    pass


gen.add_command(grover)
