# Ersilia Embeddings

A lite wrapper for Ersilia compound embeddings model.

## Quick start guide

### 1. Clone the repository

```bash
git clone https://github.com/ersilia-os/compound-embedding.git
cd compound-embedding/lite
```

### 2. Install the package with pip

```bash
pip install -e .
```

or if you have a GPU

```bash
pip install -e .[gpu]
```

### 3. Programatically generate embeddings

```python
from eosce.models import ErsiliaCompoundEmbeddings
model = ErsiliaCompoundEmbeddings()
embeddings = model.transform(["CCOC(=O)C1=CC2=CC(OC)=CC=C2O1"])
```

### 4. Generate embeddings using the cli

```bash
eosce embed "CCOC(=O)C1=CC2=CC(OC)=CC=C2O1"
```
