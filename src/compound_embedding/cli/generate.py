"""Generate datasets."""

import csv
from pathlib import Path

import click

from compound_embedding.pipelines.common import (
    parallel_on_generic,
    get_all_paths,
)
from compound_embedding.pipelines.gen_dataset import gen_training_data
from compound_embedding.pipelines.merge_grover import gen_grover_merged_files
from compound_embedding.pipelines.merge_mordred import gen_mordred_merged_files


@click.command()
@click.option("--ref", help="Path to reference file", required=True)
@click.option("--out", help="Output file path", required=True)
@click.option("--jobs", type=int, help="Number of threads to use", default=10)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
def efp(ref: str, out: str, jobs: int, seq: bool) -> None:
    """Generate training data for ersilia fingerprints."""
    ref_lib = Path(ref)
    with open(ref_lib) as f:
        file = csv.reader(f)
        smiles_list = [line[0] for line in file]

    global_index_map = {smile: index for (index, smile) in enumerate(smiles_list)}
    chunks = []
    chunk_size = len(smiles_list) // jobs
    for i in range(0, len(smiles_list), chunk_size):
        chunks.append(slice(i, i + chunk_size))
    if not seq:
        parallel_on_generic(
            smiles_list,
            gen_training_data,
            [Path(out), 1000, True, global_index_map, len(smiles_list)],
            jobs,
        )


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--jobs", type=int, help="Number of threads to use", default=10)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
@click.option(
    "--rm-inp", is_flag=True, help="Delete input data to save space while processing."
)
def grover(inp: str, out: str, jobs: int, seq: bool, rm_inp: bool) -> None:
    """Generate grover dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_grover_merged_files(file_paths, Path(out))
    else:
        parallel_on_generic(
            file_paths, gen_grover_merged_files, [Path(out), rm_inp], jobs
        )


@click.command()
@click.option("--inp", help="Input directory path", required=True)
@click.option("--out", help="Output directory path", required=True)
@click.option("--jobs", type=int, help="Number of threads to use", default=10)
@click.option("--seq", is_flag=True, help="Run pipeline sequentially")
@click.option(
    "--rm-inp", is_flag=True, help="Delete input data to save space while processing."
)
def mordred(inp: str, out: str, jobs: int, seq: bool, rm_inp: bool) -> None:
    """Generate mordred dataset."""
    file_paths = get_all_paths(Path(inp))
    if seq:
        gen_mordred_merged_files(file_paths, Path(out))
    else:
        parallel_on_generic(
            file_paths, gen_mordred_merged_files, [Path(out), rm_inp], jobs
        )


@click.group()
def gen() -> None:
    """Generate dataset commands."""
    pass


gen.add_command(grover)
gen.add_command(mordred)
