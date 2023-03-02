import csv
from pathlib import Path

import joblib
from mpi4py import MPI

from compound_embedding.pipelines.gen_dataset import gen_training_data


comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

ref_lib = Path(__file__).parent.joinpath("reference_library.csv")
with open(ref_lib) as f:
    file = csv.reader(f)
    smiles_list = [line[0] for line in file]

global_index_map = {smile: index for (index, smile) in enumerate(smiles_list)}
joblib.dump(global_index_map, "global_index_map.joblib")

chunks = []
chunk_size = len(smiles_list) // comm.size
for i in range(0, (len(smiles_list) - 1), chunk_size):
    chunks.append(slice(i, i + chunk_size))
print(chunks)
gen_training_data(
    smiles_list[chunks[rank]],
    Path("./efp_training.hdf5"),
    3000,
    True,
    global_index_map,
    len(smiles_list),
)
