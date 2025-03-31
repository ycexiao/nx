import json

import numpy as np
from monty.serialization import dumpfn, loadfn


def check_same_values(list_of_values):
    if ((np.array(list_of_values) - list_of_values[0]) == 0).all():
        return True
    else:
        return False


def query_structural_data(element, num_chunks, save_name="tmp.json.gz"):
    from mp_api.client import MPRester

    with MPRester() as mpr:
        docs = mpr.materials.search(
            elements=[element],
            num_chunks=num_chunks,
            fields=["material_id", "structure"],
        )
    dumpfn(docs, save_name)


def get_coordination_number(structure, element):
    import re

    import pymatgen

    sites = structure._sites
    coordination_numbers = []

    for i in range(len(sites)):
        if re.search(element, sites[i]._label):
            cn = pymatgen.analysis.local_env.CrystalNN().get_cn(structure, i)
            coordination_numbers.append(cn)

    return coordination_numbers


def filter_same_coordination_number(load_name, save_name, element):

    docs = loadfn(load_name)
    save_docs = []
    for i in range(len(docs)):
        coordination_numbers = get_coordination_number(docs[i]["structure"], element)
        if check_same_values(coordination_numbers):
            docs[i]["my_coordination_number"] = coordination_numbers[0]
            docs[i]["structure"] = docs[i]["structure"].as_dict()
            save_docs.append(docs[i])
        else:
            continue

    with open(save_name, "w") as f:
        json.dump(save_docs, f)
    print("Original data entries: {}".format(len(docs)))
    print("Filtered data entries: {}".format(len(save_docs)))


def main():
    element = "Fe"
    num_chunks = 1

    queried_data_path = "queried_data.json.gz"
    filtered_data_path = "filtered_data.json"

    # query_structural_data(element, num_chunks, save_name=queried_data_path)
    # filter_same_coordination_number(queried_data_path, filtered_data_path, element)
    # #  Next: try to set strict rules to get more accurate coordination number


if __name__ == "__main__":

    main()

    # for key, val in docs[0].items():
    #     print(key)
    # print(type(val))

    # help(type(docs[0]['structure']))
