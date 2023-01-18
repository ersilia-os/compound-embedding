"""Merge FS-Mol dataset with other datasets."""

from collections import OrderedDict
import gzip
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Any, List, Optional

from groverfeat import Featurizer
from joblib import cpu_count, delayed, Parallel
from mordred import Calculator, descriptors
from rdkit import Chem
from tqdm import tqdm


def get_all_paths(dir_path: Path) -> List[Path]:
    """Get all data file paths.

    Args:
        dir_path (Path): Data directory.

    Returns:
        List[Path]: Data file paths.
    """
    path_gen = Path(dir_path).glob('**/*')
    files = [x for x in path_gen if x.is_file()]
    return files


def read_as_jsonl(self, error_handling: Optional[Callable[[str, Exception], None]] = None) -> Iterable[Any]:
    """
    Parse JSONL files. See http://jsonlines.org/ for more.

    Args:
        error_handling: a callable that receives the original line and the exception object and takes
            over how parse error handling should happen.
    Returns:
        Iterable[Any]: a iterator of the parsed objects of each line.
    """
    for line in self.read_as_text().splitlines():
        try:
            yield json.loads(line, object_pairs_hook=OrderedDict)
        except Exception as e:
            if error_handling is None:
                raise
            else:
                error_handling(line, e)


def write_jsonl_gz_data(file_name: str, data: Iterable[Dict[str, Any]], len_data: int = None) -> int:
    """_summary_

    Args:
        file_name (str): _description_
        data (Iterable[Dict[str, Any]]): _description_
        len_data (int, optional): _description_. Defaults to None.

    Returns:
        int: _description_
    """
    num_ele = 0
    with gzip.open(file_name, "wt") as data_fh:
        for ele in tqdm(data, total=len_data):
            save_element(ele, data_fh)
            num_ele += 1
    return num_ele


def save_element(element: Dict[str, Any], data_fh) -> None:
    ele = dict(element)
    ele.pop("mol", None)
    ele.pop("fingerprints_vect", None)
    if "fingerprints" in ele:
        ele["fingerprints"] = ele["fingerprints"].tolist()
    data_fh.write(json.dumps(ele) + "\n")


def gen_ref_smiles_to_grover_map(ref_csvs=[]) -> Dict:
    """_summary_

    Args:
        ref_csvs (list, optional): _description_. Defaults to [].

    Returns:
        Dict: _description_
    """
    ref_map = {}
    for i, ref_csv in enumerate(ref_csvs):
        for j, smile in enumerate(ref_csv):
            ref_map[smile] = (i, j)
    return ref_map


def parallel_on_paths(file_paths: List[Path], func: Callable[[List[Path], Any]], args: List[Any] = []) -> None:
    """Execute function in parallel on multiple threads.

    Args:
        file_paths (List[Path]): List of file paths to chunk and distribute.
        func (Callable[List[Path], Any]): Processing function.
        args (List[Any]): List of additional args to the processing function.
    """
    file_path_len = len(file_paths)
    slices = []
    for i in range(0, file_path_len, file_path_len // cpu_count()):
        slices.append(slice(i, i + file_path_len // cpu_count()))

    file_path_chunks = [file_paths[s] for s in slices]
    Parallel(n_jobs=cpu_count())(
        delayed(func)(chunk, *args)
        for chunk in tqdm(file_path_chunks)
    )


def gen_grover_merged_files(task_file_paths: List[Path], output_dir: Path) -> None:
    """Merge fs-mol with grover fingerprints.

    Args:
        task_file_paths (List[Path]): List of all tasks.
        output_dir (Path): Path to save generated files
    """
    # Ensure dirs are present
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.joinpath("train"), exist_ok=True)
    os.makedirs(output_dir.joinpath("test"), exist_ok=True)
    os.makedirs(output_dir.joinpath("valid"), exist_ok=True)
    grover_model = Featurizer()
    for path in task_file_paths:
        samples = []
        sample_smiles = []
        modified_data = []
        output_file_path = output_dir.joinpath(*path.parts[-2:])
        for sample in read_as_jsonl(path):
            samples.append(sample)
            sample_smiles.append(sample["SMILES"])

        # Generate grover fingerprints in batch
        grover_fps = grover_model.transform(sample_smiles)

        # Update samples with grover data
        for i, sample in enumerate(sample_smiles):
            sample["grover"] = grover_fps[i]
            modified_data.append(sample)

        # Write modified files
        write_jsonl_gz_data(output_file_path, modified_data, len(modified_data))


def gen_mordred_merged_files(task_file_paths: List[Path], output_dir: Path) -> None:
    """Merge fs-mol with mordred descriptors.

    Args:
        task_file_paths (List[Path]): List of all tasks.
        output_dir (Path): Path to save generated files
    """
    # Ensure dirs are present
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.joinpath("train"), exist_ok=True)
    os.makedirs(output_dir.joinpath("test"), exist_ok=True)
    os.makedirs(output_dir.joinpath("valid"), exist_ok=True)
    mordred_calc = Calculator(descriptors)
    for path in task_file_paths:
        modified_data = []
        output_file_path = output_dir.joinpath(*path.parts[-2:])
        for sample in read_as_jsonl(path):
            sample["mordred"] = mordred_calc(Chem.MolFromSmiles(sample["SMILES"]))
            modified_data.append(sample)
        write_jsonl_gz_data(output_file_path, modified_data, len(modified_data))
