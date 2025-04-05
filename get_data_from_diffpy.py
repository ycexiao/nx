import json

from diffpy.srreal.parallel import createParallelCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.structure import loadStructure
from diffpy.structure.parsers import getParser
from matplotlib import pyplot as plt
from pymatgen.core import Structure
from pyobjcryst import loadCrystal

import numpy as np


def get_diffpy_structure(structure_dict):
    """Place to set constraints on the structure
    """
    structure_pymatgen = Structure.from_dict(structure_dict)  # get from dict
    structure_cif = structure_pymatgen.to(fmt="cif")  # dict to cif
    parser = getParser(format="cif")  
    structure_diffpy = parser.parse(structure_cif)  # cif to diffpy.structure
    structure_diffpy.Uisoequiv = 0.007  # setting structure values
    return structure_diffpy


def get_pdf(structure_diffpy, mode='both', u=0.007, rmax=10.0, qmax=30.0):
    """Place to set constraints on the pdf calculator
    """
    structure_diffpy.Uisoequiv = u  # set the isotropic thermal parameter
    xray_pdf_calculator = PDFCalculator(rmax=rmax, qmax=qmax)
    x_pdf = xray_pdf_calculator(structure_diffpy)

    if mode=='both':
        neutron_pdf_calculator = PDFCalculator(rmax=rmax, qmax=qmax)
        neutron_pdf_calculator.scatteringfactortable = "neutron"
        n_pdf = neutron_pdf_calculator(structure_diffpy)
        return x_pdf, n_pdf
    elif mode == 'xray':
        return x_pdf

def sample_pdf(pdf, offset = 0, qmax=30, rmax=10,verbose=False):
    expected_length = int(np.ceil(qmax * rmax / np.pi)+1) 
    input_length = len(pdf[0])
    expected_length_round = (expected_length // 10 -expected_length // 100 * 10 + 1) * 10 +expected_length // 100 * 100
    length =expected_length_round + offset
    if verbose:
        print(f"Orignial PDF length: {input_length},\n"
              f"Sampled PDF length: {expected_length_round},\n"
              f"Use PDF length: {length}")
    
    x_new = np.linspace(pdf[0][0], pdf[0][-1], length)
    n_pdf_new = np.interp(x_new, pdf[0], pdf[1], left=0, right=0)
    return list(x_new), list(n_pdf_new)
    
        
    

def quick_plot_pdf(x_pdf, n_pdf):
    fig, ax = plt.subplots()
    ax.plot(x_pdf[0], x_pdf[1])
    ax.plot(n_pdf[0], n_pdf[1])
    plt.show()
    return fig, ax


def add_pdf_to_datasets(load_name, save_name):
    with open(load_name, "r") as f:
        docs = json.load(f)

    for doc in docs:
        #        print(doc['formula_pretty'])
        #        print(doc['my_coordination_number'])
        structure = get_diffpy_structure(doc["structure"])
        x_pdf, n_pdf, _ = get_pdf(structure)
        doc["x_pdf"] = list(x_pdf)
        doc["n_pdf"] = list(n_pdf)

    with open(save_name, "w") as f:
        json.dump(docs, f, indent=4)


if __name__ == "__main__":

    filtered_data_path = "filtered_data.json"
    complete_data_path = "complete_data.json"
#add_pdf_to_datasets(filtered_data_path, complete_data_path)


    with open(filtered_data_path, "r") as f:
        docs = json.load(f)
    structure = get_diffpy_structure(docs[0]["structure"])
    x_pdf, n_pdf = get_pdf(structure)
    quick_plot_pdf(x_pdf, n_pdf)

