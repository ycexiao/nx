import json
import sys
from matplotlib import pyplot as plt

# runing in the project root
import sys
sys.path.append(os.path.abspath(os.getcwd()))
from get_data_from_diffpy import get_pdf_structure, get_pdf, quick_plot_pdf


load_name = "playground/pairs/queried_data.json.gz"
with open(load_name, 'r') as f:
    docs = json.load(f)

for doc in docs:
    structure = get_diffpy_structure(doc["structure"])
    x_pdf, n_pdf = get_pdf(structure)
    doc['x_pdf'] = [list(x_pdf[i]) for i in range(len(x_pdf))]
    doc['n_pdf'] = [list(n_pdf[i]) for i in range(len(n_pdf))]

save_name = 'playground/pairs/complete_data.json'
with open(save_name, 'w') as f:
    json.dump(docs, f, indent=4)