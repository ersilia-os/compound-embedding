"""Models."""

from typing import List, Optional
from pathlib import Path

import numpy as np
import onnxruntime as ort

from eosce.utils import get_package_root_path, smiles_to_morgan

class ErsiliaCompoundEmbeddings:
    def __init__(self: "ErsiliaCompoundEmbeddings", onnx_path: Optional[Path] = get_package_root_path().joinpath("efp.onnx")) -> None:
        """Initialize Ersilia compound embeddings model.

        Args:
            onnx_path (Optional[Path], optional): Path to model onnx file.
                Defaults to get_package_root_path().joinpath("efp.onnx").
        """
        
        self.providers = ort.get_available_providers()
        print(f"Available providers: {self.providers}")
        self.session = ort.InferenceSession(str(onnx_path), provider_options=self.providers[0])

    def transform(self: "ErsiliaCompoundEmbeddings", smiles: List[str]) -> List[np.ndarray]:
        """Transform smiles to embeddings.

        Args:
            smiles (List[str]): A list of smile strings.

        Returns:
            List[np.ndarray]: A list of embeddings.
        """
        # Convert smiles to morgan
        morgan = smiles_to_morgan(smiles)
        output = self.session.run(["embeddings_list"], {"smiles_list": np.asarray(morgan, dtype=np.float32)})
        return output
