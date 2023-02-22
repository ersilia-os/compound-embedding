"""Train models."""

import os
import time
from pathlib import Path

import click
from dpu_utils.utils import RichPath
import torch
import joblib

from compound_embedding.fs_mol.data.fsmol_dataset import FSMolDataset
from compound_embedding.fs_mol.models.abstract_torch_fsmol_model import (
    resolve_starting_model_file,
)
from compound_embedding.fs_mol.utils.protonet_utils import (
    evaluate_protonet_model,
    PrototypicalNetworkTrainer,
)
from compound_embedding.fs_mol.utils.cli_utils import set_seed


@click.command()
@click.option(
    "--save_dir", type=str, help="Directory to save model and run info.", required=True
)
@click.option(
    "--data_dir", type=str, help="Directory to load test data.", required=True
)
@click.option("--model_path", type=str, help="Model checkpoint path.", required=True)
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
    model_path: str,
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
    """Test Protonet."""

    set_seed(seed, True, False)
    # load dataset
    dataset = FSMolDataset.from_directory(
        directory=RichPath.create(data_dir), num_workers=5
    )

    run_name = f"FSMol_Test_protonet_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = Path(save_dir).joinpath(run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weights_file = resolve_starting_model_file(
        model_file=model_path,
        model_cls=PrototypicalNetworkTrainer,
        out_dir=out_dir,
        use_fresh_param_init=False,
        device=device,
    )

    model = PrototypicalNetworkTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )
    print(f"\tDevice: {device}")
    print(f"\tNum parameters {sum(p.numel() for p in model.parameters())}")
    print(f"\tModel:\n{model}")

    results = evaluate_protonet_model(
        model,
        dataset,
        save_dir=out_dir,
        support_sizes=[16, 32, 64, 128, 256],
        seed=seed,
    )
    joblib.dump(results, "test_results_protonet.joblib")


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

    run_name = f"EFP_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir = Path(save_dir).joinpath(run_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {}
    model.to(device)


@click.group()
def test() -> None:
    """Test model commands."""
    pass


test.add_command(protonet)
test.add_command(efp)
