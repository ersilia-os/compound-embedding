"""Train models."""

import os
import time
from pathlib import Path

import click
from dpu_utils.utils import RichPath
import torch

from compound_embedding.fs_mol.data.fsmol_dataset import FSMolDataset
from compound_embedding.fs_mol.utils.protonet_utils import (
    PrototypicalNetworkTrainerConfig,
    PrototypicalNetworkTrainer,
)
from compound_embedding.fs_mol.utils.cli_utils import set_seed


@click.command()
@click.option(
    "--save_dir", type=str, help="Directory to save model and run info.", required=True
)
@click.option("--data_dir", type=str, help="Dataset directory.", required=True)
@click.option(
    "--features",
    type=click.Choice(["ecfp+grover+mordred+fc", "ecfp+grover+fc"]),
    help="Choice of features to use",
    default="ecfp+grover+mordred+fc",
)
@click.option(
    "--distance_metric",
    type=click.Choice(["mahalanobis", "euclidean"]),
    help="Choice of distance to use.",
    default="euclidean",
)
@click.option("--support_set_size", type=int, default=64, help="Size of support set")
@click.option(
    "--query_set_size",
    type=int,
    default=256,
    help="Size of target set. If -1, use everything but train examples.",
)
@click.option(
    "--tasks_per_batch",
    type=int,
    default=16,
    help="Number of tasks to accumulate gradients for.",
)
@click.option(
    "--batch_size", type=int, default=256, help="Number of examples per batch."
)
@click.option(
    "--num_train_steps", type=int, default=10000, help="Number of training steps."
)
@click.option("--lr", type=float, default=0.0001, help="Learning rate")
@click.option(
    "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
)
@click.option("--seed", type=int, help="Set random seed.", default=42)
def protonet(
    save_dir: str,
    data_dir: str,
    features: str,
    distance_metric: str,
    support_set_size: int,
    query_set_size: int,
    tasks_per_batch: int,
    batch_size: int,
    num_train_steps: int,
    lr: float,
    clip_value: float,
    seed: int,
) -> None:
    """Generate grover dataset."""
    config = PrototypicalNetworkTrainerConfig(
        used_features=features,
        distance_metric=distance_metric,
        batch_size=batch_size,
        tasks_per_batch=tasks_per_batch,
        support_set_size=support_set_size,
        query_set_size=query_set_size,
        validate_every_num_steps=50,
        validation_support_set_sizes=tuple([16, 128]),
        validation_query_set_size=512,
        validation_num_samples=5,
        num_train_steps=num_train_steps,
        learning_rate=lr,
        clip_value=clip_value,
    )

    set_seed(seed, True, False)

    # load dataset
    dataset = FSMolDataset.from_directory(directory=RichPath.create(data_dir))

    run_name = f"FSMol_protonet_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = Path(save_dir).joinpath(run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = PrototypicalNetworkTrainer(config=config).to(device)

    print(f"\tDevice: {device}")
    print(f"\tNum parameters {sum(p.numel() for p in model_trainer.parameters())}")
    print(f"\tModel:\n{model_trainer}")

    model_trainer.train_loop(out_dir, dataset, device)


@click.command()
@click.option(
    "--save_dir", type=str, help="Directory to save model and run info.", required=True
)
@click.option("--data_dir", type=str, help="Dataset directory.", required=True)
@click.option(
    "--batch_size", type=int, default=256, help="Number of examples per batch."
)
@click.option(
    "--num_train_steps", type=int, default=10000, help="Number of training steps."
)
@click.option("--lr", type=float, default=0.0001, help="Learning rate")
@click.option("--seed", type=int, help="Set random seed.", default=42)
def efp(
    save_dir: str,
    data_dir: str,
    batch_size: int,
    num_train_steps: int,
    lr: float,
    seed: int,
) -> None:
    set_seed(seed, True, False)
    # load dataset

    run_name = f"FSMol_protonet_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = Path(save_dir).joinpath(run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {}
    model.to(device)
    model.train()


@click.group()
def train() -> None:
    """Train model commands."""
    pass


train.add_command(protonet)
train.add_command(efp)
