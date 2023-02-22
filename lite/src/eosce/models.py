"""Models."""

import logging
from typing import Any, List, Optional
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eosce.utils import get_package_root_path, smiles_to_morgan


def get_ort_session(onnx_path: Path, providers: Optional[List[str]] = ort.get_available_providers()) -> Any:
    """Create an inference session with working provider.

    Args:
        onnx_path (Path): Path to model onnx file.
        providers (Optional[List[str]], optional): _description_. Defaults to ort.get_available_providers().

    Returns:
        Any: ORT Inference session.
    """
    logging.debug(f"Available providers: {providers}")
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return get_ort_session(onnx_path, providers[1:])


class ErsiliaCompoundEmbeddings:
    def __init__(self: "ErsiliaCompoundEmbeddings", onnx_path: Optional[Path] = get_package_root_path().joinpath("efp.onnx")) -> None:
        """Initialize Ersilia compound embeddings model.

        Args:
            onnx_path (Optional[Path], optional): Path to model onnx file.
                Defaults to get_package_root_path().joinpath("efp.onnx").
        """
        self.session = get_ort_session(onnx_path)            

    def transform(self: "ErsiliaCompoundEmbeddings", smiles: List[str]) -> np.ndarray:
        """Transform smiles to embeddings.

        Args:
            smiles (List[str]): A list of smile strings.

        Returns:
            np.ndarray: An array of embeddings.
        """
        # Convert smiles to morgan
        morgan_list = smiles_to_morgan(smiles)
        embedding_list = self.session.run(["embedding_list"], {"morgan_list": np.asarray(morgan_list, dtype=np.float32)})
        return embedding_list[0]
