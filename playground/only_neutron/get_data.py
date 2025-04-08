from get_data_from_diffpy import get_pdf, get_diffpy_structure, sample_pdf
from pymatgen.core.structure import Structure
import os
import json
import numpy as np



## generate example dataset with neutron PDF

dir_path = '../../example_datasets'
elements = ['Ti', "Fe", "Mn", 'Cu']
file_path = [element + '_collection.json' for element in elements] # filename or path to the collection
file_path = [os.path.join(dir_path, file_path[i]) for i in range(len(file_path))]

dump_file = 'playground/only_neutron/neutron_collection.json'


with open(file_path[0], 'r') as file:
    example_collection = json.load(file)

for i in range(len(example_collection)):
    stru = get_diffpy_structure(example_collection[i]['structure'])
    pdf = get_pdf(stru, mode='xray')
    example_collection[i]['npdf'] = sample_pdf(pdf, offset=100)


with open( dump_file, 'w') as dump:
    # Dump the modified collection with neutron PDF data
    json.dump(example_collection, dump, indent=4)
