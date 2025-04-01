import json
from matplotlib import pyplot as plt

from pyobjcryst import loadCrystal
from diffpy.srreal.parallel import createParallelCalculator
from diffpy.structure.parsers import getParser
from pymatgen.core import Structure
from diffpy.structure import loadStructure
from diffpy.srreal.pdfcalculator import PDFCalculator


def get_diffpy_structure(structure_dict):
    structure_pymatgen = Structure.from_dict(structure_dict)
    structure_cif = structure_pymatgen.to(fmt="cif")
    parser = getParser(format="cif")
    structure_diffpy = parser.parse(structure_cif)
    structure_diffpy.Bisoequiv = 0.5
    return structure_diffpy

def get_pdf(structure_diffpy):
    xray_pdf_calculator = PDFCalculator(qmin=0, qmax=10)
    neutron_pdf_calculator = PDFCalculator(qmin=0, qmax=10)
    neutron_pdf_calculator.scatteringfactortable = 'neutron'
    x_pdf = xray_pdf_calculator(structure_diffpy)
    n_pdf = neutron_pdf_calculator(structure_diffpy)
    return x_pdf[1], n_pdf[1], x_pdf[0]

def quick_plot_pdf(x_pdf, n_pdf):
    fig ,ax = plt.subplots(1,2)
    ax[0].plot(x_pdf[0], x_pdf[1])
    ax[1].plot(n_pdf[0], n_pdf[1])
    plt.show()
    return fig, ax

def add_pdf_to_datasets(load_name, save_name):
    with open(load_name, "r") as f:
        docs = json.load(f)

    for doc in docs:
#        print(doc['formula_pretty'])
#        print(doc['my_coordination_number'])
        structure = get_diffpy_structure(doc["structure"])
        x_pdf, n_pdf,_ = get_pdf(structure)
        doc['x_pdf'] = list(x_pdf)
        doc['n_pdf'] = list(n_pdf)

    with open(save_name, "w") as f:
        json.dump(docs, f, indent=4)
    


if __name__ == "__main__": 


    filtered_data_path = "filtered_data.json"
    complete_data_path = 'complete_data.json'

    add_pdf_to_datasets(filtered_data_path, complete_data_path)
