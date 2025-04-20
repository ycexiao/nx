""" Example datasets have only selected fields. Need complete fields to investigate model"""

import json
import numpy as np
import os
from mp_api.client import MPRester

elements = ['Ti', "Fe", "Mn", 'Cu']
file_names = [element + '_collection.json' for element in elements] # filename or path to the collection

example_dir = 'example_datasets'
example_path = [os.path.join(example_dir, file_names[i]) for i in range(len(file_names))]

dump_dir = 'complete_example_datasets'
dump_path = [os.path.join(dump_dir, file_names[i]) for i in range(len(file_names))]

with open(example_path[0], 'r') as file:
    collection = json.load(file)

print(collection[0]['structure'])


# for i in range(len(file_names)):
#     with open(example_path[i], 'r') as file:
#         collection = json.load(file)

#     ids = [collection[j]['mp_id'] for j in range(len(collection))]

#     with MPRester() as mpr:
#         docs = mpr.materials.search(material_ids = ids)
    
#     for k in range(len(docs)):
#         docs[k] = docs[k].dict()  
    
#     with open(dump_path[i], 'w') as dump:
#         # Dump the modified collection with neutron PDF data
#         json.dump(docs, dump, indent=4)


