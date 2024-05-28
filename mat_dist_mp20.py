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

#%%
# load data
diffcsp_dir = './'
datadir = join(diffcsp_dir, 'data/mp_20')   #!
file =  join(datadir, 'test.csv')
print("datadir: ", datadir)
# Load Strucutres from csv file (either from train, val, test)
df0 = pd.read_csv(file)
# randompy sample 2000 datapoints
df0 = df0.sample(5000).reset_index(drop=True)
pstruct_dict_db = {df0['material_id'][i]: str2pmg(df0['cif'][i]) for i in range(len(df0))}
pstruct_list_db = list(pstruct_dict_db.values())
mpids_db = list(pstruct_dict_db.keys())

#%%
with MPRester(API_KEY) as mpr:

    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
        stats=('mean', 'std_dev', 'minimum', 'maximum')
    )
    
    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Get structures.
    for i, (mpid, pstruct) in tqdm(enumerate(pstruct_dict_db.items()), total=len(pstruct_dict_db)):
        try:
            v_cnn = np.array(ssf.featurize(pstruct))
            if v_cnn is None or np.isnan(v_cnn).any():
                print(f"Skipping mpid {mpid} due to None or NaN in v_cnn")
                continue

            # get composition
            comp = pstruct.composition
            comp_feature = generate_magpie_features(str(comp))
            if comp_feature is None or pd.isna(comp_feature).any():
                print(f"Skipping mpid {mpid} due to None or NaN in comp_feature")
                continue

            sga = SpacegroupAnalyzer(pstruct)
            sg = sga.get_space_group_number()
            
            # get the lattice type like 'cubic', 'hexagonal', etc.
            lattice_type = sga.get_lattice_type()

            # store all info as pandas dataframe
            new_row = pd.DataFrame({
                'mpid': [mpid], 
                'composition': [comp], 
                'spacegroup': [sg], 
                'lattice_type': [lattice_type],
                'cnn_features': [v_cnn], 
                'magpie_features': [comp_feature]
            })
            df = pd.concat([df, new_row], ignore_index=True)
        except Exception as e:
            print(f"Error: {e}")
            continue

#%%
# PCA analysis for data in df['cnn_features']
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Standardize the data
scaler = StandardScaler()
X = np.vstack(df['cnn_features'])
X = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(10, 10))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['spacegroup'], palette='viridis')
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['lattice_type'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of CNN features')
plt.show()

#%%
# PCA analysis for data in df['magpie_features']

# Standardize the data
scaler = StandardScaler()
X = np.vstack(df['magpie_features'])
X = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot
plt.figure(figsize=(10, 10))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['spacegroup'], palette='viridis')
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['lattice_type'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of CNN features')
plt.show()

#%%
# concatenate cnn_features and magpie_features, then do pca analysis
X = np.vstack(df['cnn_features'])
X = scaler.fit_transform(X)
Y = np.vstack(df['magpie_features'])
Y = scaler.fit_transform(Y)
Z = np.hstack((X, Y))

# PCA
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z)

# Plot
plt.figure(figsize=(10, 10))
# sns.scatterplot(x=Z_pca[:, 0], y=Z_pca[:, 1], hue=df['spacegroup'], palette='viridis')
sns.scatterplot(x=Z_pca[:, 0], y=Z_pca[:, 1], hue=df['lattice_type'], palette='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of CNN and Magpie features')
plt.show()



# %%
