import json

from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.structure.parsers import getParser
from pymatgen.core import Structure


def get_diffpy_structure(structure_dict):
    structure_pymatgen = Structure.from_dict(structure_dict)
    structure_cif = structure_pymatgen.to(fmt="cif")
    parser = getParser(format="cif")
    structure_diffpy = parser.parse(structure_cif)
    return structure_diffpy


if __name__ == "__main__":

    filtered_data_path = "filtered_data.json"

    with open(filtered_data_path, "r") as f:
        docs = json.load(f)

    pdfc = PDFCalculator(qmax=18, rmax=10)
    qmin = 0

    for doc in docs:
        structure = get_diffpy_structure(doc["structure"])
        r0, g0 = pdfc(structure, qmin=qmin)
