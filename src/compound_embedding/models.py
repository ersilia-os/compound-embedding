"""Fingerprint models."""

from typing import Any

import torch.nn as nn


class ErsiliaFingerprint(nn.Module):
    """Ersilia fingerprint."""

    def __init__(self: "ErsiliaFingerprint"):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2048, 1706),
            nn.ReLU(),
            nn.Linear(1706, 1365),
            nn.ReLU(),
            nn.Linear(1365, 1024),
        )

    def forward(self: "ErsiliaFingerprint", input_batch: Any) -> Any:

        return self.fc(input_batch)
