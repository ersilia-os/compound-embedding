"""Common pipeline utils."""

from collections import OrderedDict
import gzip
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Any, List, Optional

from joblib import cpu_count, delayed, Parallel
import numpy as np
from tqdm import tqdm


def get_all_paths(dir_path: Path) -> List[Path]:
    """Get all data file paths.

    Args:
        dir_path (Path): Data directory.

    Returns:
        List[Path]: Data file paths.
    """
    path_gen = Path(dir_path).glob("**/*")
    files = [x for x in path_gen if x.is_file()]
    return files


def get_package_root_path() -> Path:
    """Get path of the package root.

    Returns:
        Path: Package root path
    """
    return Path(__file__).parents[1].absolute()


def read_as_jsonl(
    path, error_handling: Optional[Callable[[str, Exception], None]] = None
) -> Iterable[Any]:
    """
    Iterate through JSONL files. See http://jsonlines.org/ for more.

    :param error_handling: a callable that receives the original line and the exception object and takes
            over how parse error handling should happen.
    :return: a iterator of the parsed objects of each line.
    """
    fh = gzip.open(path, mode="rt", encoding="utf-8")
    try:
        for line in fh:
            try:
                yield json.loads(line, object_pairs_hook=OrderedDict)
            except Exception as e:
                if error_handling is None:
                    raise
                else:
                    error_handling(line, e)
    finally:
        fh.close()


def write_jsonl_gz_data(
    file_name: str, data: Iterable[Dict[str, Any]], len_data: int = None
) -> int:
    """_summary_

    Args:
        file_name (str): _description_
        data (Iterable[Dict[str, Any]]): _description_
        len_data (int): _description_. Defaults to None.

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
    if "grover" in ele:
        ele["grover"] = [
            str(e) for e in np.array(ele["grover"], dtype=np.float16).tolist()
        ]
    if "mordred" in ele:
        ele["mordred"] = [
            str(e) for e in np.array(ele["mordred"], dtype=np.float16).tolist()
        ]
    data_fh.write(json.dumps(ele) + "\n")


def parallel_on_generic(
    generic_iter: Iterable[Any],
    func: Callable[[List[Path], Any], Any],
    args: List[Any] = [],
    jobs: int = None,
) -> None:
    """Execute function in parallel on multiple threads.

    Args:
        generic_iter (Iterable[Any]): A generic iterable to chunk and distribute.
        func (Callable[List[Path], Any]): Processing function.
        args (List[Any]): List of additional args to the processing function.
        jobs (int): Number of cpu cores to use.
    """
    jobs = jobs or cpu_count()
    generic_iter_len = len(generic_iter)
    slices = []
    for i in range(0, generic_iter_len, generic_iter_len // jobs):
        slices.append(slice(i, i + generic_iter_len // jobs))

    generic_iter_chunks = [generic_iter[s] for s in slices]
    Parallel(n_jobs=jobs)(
        delayed(func)(chunk, *args) for chunk in tqdm(generic_iter_chunks)
    )
