from src.eosce.models import ErsiliaCompoundEmbeddings
import sys

print(sys.argv[1])

model = ErsiliaCompoundEmbeddings()
embeddings = model.transform(sys.argv[1])