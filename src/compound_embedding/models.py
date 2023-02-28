"""Fingerprint models."""

from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py as h5
import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch.nn import functional as F
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchmetrics import MeanSquaredError, MetricCollection

from compound_embedding.fs_mol.utils.torch_utils import torchify


def to_tensor(
    sample: Dict[str, np.ndarray], device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sample to tensor values."""
    return tuple(
        [torchify(np.asarray(val, dtype=np.float32), device) for val in sample.values()]
    )


def read_datapoint(
    index: int,
    file_path: Path,
    field_names: List[str],
) -> Dict:
    h5_file = h5.File(file_path, mode="r")
    return {dataset: h5_file[dataset][index] for dataset in field_names}


def build_datapipe(
    indices_range: List, data_file: Path, field_names: List[str], device: str = "cpu"
) -> IterDataPipe:
    part_to_tensor = partial(to_tensor, device=device)
    part_read_datapoint = partial(
        read_datapoint, file_path=data_file, field_names=field_names
    )
    return (
        IterableWrapper(range(indices_range[0], indices_range[1]))
        .shuffle(buffer_size=(indices_range[1] - indices_range[0]))
        .sharding_filter()
        .map(fn=part_read_datapoint)
        .map(fn=part_to_tensor)
    )


class ErsiliaFingerprint(LightningModule):
    """Ersilia fingerprint."""

    def __init__(self: "ErsiliaFingerprint"):
        super().__init__()

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1840),
            torch.nn.ReLU(),
            torch.nn.Linear(1840, 1636),
            torch.nn.ReLU(),
            torch.nn.Linear(1636, 1432),
            torch.nn.ReLU(),
            torch.nn.Linear(1432, 1228),
            torch.nn.ReLU(),
            torch.nn.Linear(1228, 1024),
        )

        metrics = MetricCollection([MeanSquaredError()])
        self.train_metrics = metrics.clone(prefix="TRAIN_")
        self.valid_metrics = metrics.clone(prefix="VAL_")
        self.test_metrics = metrics.clone(prefix="TEST_")

        self.criterion = F.mse_loss

    def forward(self: "ErsiliaFingerprint", input_batch: Any) -> Any:
        embedding = self.fc(input_batch)
        return embedding

    def training_step(self: "ErsiliaFingerprint", batch: Any, batch_idx: Any) -> Any:
        """Training step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index

        Returns:
            Any: training step loss
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("TRAIN_Loss", loss)

        # Logging metrics
        metrics = self.train_metrics(output, y)
        self.log_dict(metrics)
        return loss

    def validation_step(self: "ErsiliaFingerprint", batch: Any, batch_idx: Any) -> Any:
        """Validation step.

        Args:
            batch (Any): input batch
            batch_idx (Any): batch index
        """
        x, y = batch
        output = self(x)
        loss = self.criterion(output, y)

        # Logging to TensorBoard by default
        self.log("VAL_Loss", loss)

        # Logging metrics
        metrics = self.valid_metrics(output, y)
        self.log_dict(metrics)

    def configure_optimizers(self: "ErsiliaFingerprint") -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
