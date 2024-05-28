#%%
import numpy as np
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import os
from os.path import join
from tqdm import tqdm
import pandas as pd
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition
from sklearn.metrics.pairwise import cosine_similarity
API_KEY = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"
# sys.path.append('../')
# sys.path.append('../../')
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.composition import ElementProperty

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def str2pmg(cif):
    pstruct=Structure.from_str(cif, "CIF")
    return pstruct

def generate_magpie_features(formula):
    """
    Generate Magpie features for a given material formula.
    
    Args:
    - formula (str): Chemical formula of the material.
    
    Returns:
    - np.array: Array of Magpie features.
    """
    # Initialize the Magpie feature generator
    ep = ElementProperty.from_preset(preset_name="magpie")
    
    # Create a Composition object
    comp = Composition(formula)
    
    # Generate features
    features = ep.featurize(comp)
    
    return np.array(features).reshape(1, -1)

def calculate_similarity(features1, features2):
    """
    Calculate the cosine similarity between two feature vectors.
    
    Args:
    - features1 (np.array): Feature vector for material 1.
    - features2 (np.array): Feature vector for material 2.
    
    Returns:
    - float: Cosine similarity score.
    """
    return cosine_similarity(features1, features2)[0][0]


mpids = ['mp-2534', 'mp-2172', "mp-66", 'mp-20536', "mp-22862","mp-5827", "mp-560842"]

#%%
with MPRester(API_KEY) as mpr:

    # Calculate structure fingerprints.
    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
        stats=('mean', 'std_dev', 'minimum', 'maximum'))

    cnn_dict = {}
    comp_dict = {}
    for mpid in mpids:
        pstruct = mpr.get_structure_by_material_id(mpid)


        try:
            v_cnn = np.array(ssf.featurize(pstruct))
            if v_cnn is None or np.isnan(v_cnn).any():
                print(f"Skipping mpid {mpid} due to None or NaN in v_cnn")
                continue
            cnn_dict[mpid] = v_cnn
            # get composition
            comp = pstruct.composition
            comp_feature = generate_magpie_features(str(comp))
            if comp_feature is None or pd.isna(comp_feature).any():
                print(f"Skipping mpid {mpid} due to None or NaN in comp_feature")
                continue
            comp_dict[mpid] = comp_feature[0]

        except Exception as e:
            print(f"Error: {e}")
            continue

    # plot the confusion matrix with respect to the distance between the structures 
    dist_cnn = np.array(list(cnn_dict.values()))
    dist_cnn = np.linalg.norm(dist_cnn - dist_cnn[:, np.newaxis], axis=-1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_cnn, annot=True, fmt=".2f", cmap='viridis', cbar=False)
    plt.xlabel('Structure 1')
    plt.ylabel('Structure 2')
    plt.xticks(range(len(cnn_dict.keys())), list(cnn_dict.keys()), rotation=45)
    plt.yticks(range(len(cnn_dict.keys())), list(cnn_dict.keys()), rotation=45)

    plt.title('Distance between structures')
    plt.show()
    
    dist_comp = np.array(list(comp_dict.values()))
    dist_comp = np.linalg.norm(dist_comp - dist_comp[:, np.newaxis], axis=-1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_comp, annot=True, fmt=".2f", cmap='viridis', cbar=False)
    plt.xlabel('Structure 1')
    plt.ylabel('Structure 2')
    plt.xticks(range(len(comp_dict.keys())), list(comp_dict.keys()), rotation=45)
    plt.yticks(range(len(comp_dict.keys())), list(comp_dict.keys()), rotation=45)

    plt.title('Distance between compositions')
    plt.show()
    
# %%
