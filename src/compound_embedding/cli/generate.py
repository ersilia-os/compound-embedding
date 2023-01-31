"""Generate datasets."""

from pathlib import Path

import click

from compound_embedding.pipelines.common import (
    parallel_on_generic,
    get_all_paths,
)
from compound_embedding.pipelines.merge_grover import gen_grover_merged_files
from compound_embedding.pipelines.merge_mordred import gen_mordred_merged_files


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--jobs", type=int, help="Number of threads to use", default=10)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
@click.option("--rm-inp", is_flag=True, help="Delete input data to save space while processing.")
def grover(inp: str, out: str, jobs: int, seq: bool, rm_inp: bool) -> None:
    """Generate grover dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_grover_merged_files(file_paths, Path(out))
    else:
        parallel_on_generic(file_paths, gen_grover_merged_files, [Path(out), rm_inp], jobs)


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--jobs", type=int, help="Number of threads to use", default=10)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
@click.option("--rm-inp", is_flag=True, help="Delete input data to save space while processing.")
def mordred(inp: str, out: str, jobs: int, seq: bool, rm_inp: bool) -> None:
    """Generate mordred dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_mordred_merged_files(file_paths, Path(out))
    else:
        parallel_on_generic(file_paths, gen_mordred_merged_files, [Path(out), rm_inp], jobs)


@click.group()
def gen() -> None:
    """Generate dataset commands."""
    pass


gen.add_command(grover)
gen.add_command(mordred)
