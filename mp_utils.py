# use py311
from mp_api.client import MPRester
from pymatgen.analysis.local_env import CrystalNN
from config_file import *

def get_mpstruct(mpid):
    # Initialize the MPRester
    with MPRester(api_key) as mpr:
        # Get the structure for the specific MPID
        structure = mpr.get_structure_by_material_id(mpid)

        # Check if the structure is found
        if structure:
            # Save the structure to a dictionary
            mpdata = {str(mpid): structure}


            print(f"Structure for MPID '{mpid}' was found.")
        else:
            print(f"Structure for MPID '{mpid}' not found.")
    return structure