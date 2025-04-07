
from mp_api.client import MPRester
import json

element = ['Fe', 'Cl']
num_chunks = 1
save_name = "queried_data.json.gz"

with MPRester() as mpr:
    docs = mpr.materials.search(
        elements=[*element],
        num_elements= 2,
        num_chunks=num_chunks,        
    )


for i, doc in enumerate(docs):
    docs[i] = doc.dict()  # Convert to dict if not already

with open(save_name, "w") as f:
    json.dump(docs, f, indent=4)