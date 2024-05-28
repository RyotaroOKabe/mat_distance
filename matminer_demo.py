#%%
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from mp_utils import *

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

#%%
# Material formulas
# material_1 = "TiO2"
# material_2 = "ZrO2"
structure1, structure2 = [get_mpstruct(mpid) for mpid in ['mp-2534', 'mp-22172']]
comp1, comp2 = [s.composition for s in [structure1, structure2]]

# Generate Magpie features
# features_1 = generate_magpie_features(material_1)
# features_2 = generate_magpie_features(material_2)
features_1 = generate_magpie_features(str(comp1))
features_2 = generate_magpie_features(str(comp2))

# Calculate similarity
similarity_score = calculate_similarity(features_1, features_2)

print(f"Similarity between material_1 and material_2: {similarity_score:.3f}")

# %%
