#%%
import numpy as np
from mp_api.client import MPRester
import matplotlib.pyplot as plt
import seaborn as sns
import os
from os.path import join
import pandas as pd
API_KEY = "PvfnzQv5PLh4Lzxz1pScKnAtcmmWVaeU"
# sys.path.append('../')
# sys.path.append('../../')
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint

#%%
with MPRester(API_KEY) as mpr:

    # Get structures.
    diamond = mpr.get_structure_by_material_id("mp-66")
    gaas = mpr.get_structure_by_material_id("mp-2534")
    rocksalt = mpr.get_structure_by_material_id("mp-22862")
    perovskite = mpr.get_structure_by_material_id("mp-5827")
    # spinel_caco2s4 = mpr.get_structure_by_material_id("mvc-12728")
    spinel_sicd2O4 = mpr.get_structure_by_material_id("mp-560842")

    # Calculate structure fingerprints.
    ssf = SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0),
        stats=('mean', 'std_dev', 'minimum', 'maximum'))
    v_diamond = np.array(ssf.featurize(diamond))
    v_gaas = np.array(ssf.featurize(gaas))
    v_rocksalt = np.array(ssf.featurize(rocksalt))
    v_perovskite = np.array(ssf.featurize(perovskite))
    # v_spinel_caco2s4 = np.array(ssf.featurize(spinel_caco2s4))
    v_spinel_sicd2O4 = np.array(ssf.featurize(spinel_sicd2O4))

    # Print out distance between structures.
    print('Distance between diamond and GaAs: {:.4f}'.format(np.linalg.norm(v_diamond - v_gaas)))
    print('Distance between diamond and rocksalt: {:.4f}'.format(np.linalg.norm(v_diamond - v_rocksalt)))
    print('Distance between diamond and perovskite: {:.4f}'.format(np.linalg.norm(v_diamond - v_perovskite)))
    print('Distance between rocksalt and perovskite: {:.4f}'.format(np.linalg.norm(v_rocksalt - v_perovskite)))
    # print('Distance between Ca(CoS2)2-spinel and Si(CdO2)2-spinel: {:.4f}'.format(np.linalg.norm(v_spinel_caco2s4 - v_spinel_sicd2O4)))
    
    # plot the confusion matrix with respect to the distance between the structures 
    dist_mat = np.array([v_diamond, v_gaas, v_rocksalt, v_perovskite, v_spinel_sicd2O4])
    dist_mat = np.linalg.norm(dist_mat - dist_mat[:, np.newaxis], axis=-1)
    plt.figure(figsize=(10, 10))
    sns.heatmap(dist_mat, annot=True, fmt=".2f", cmap='viridis', cbar=False)
    plt.xlabel('Structure 1')
    plt.ylabel('Structure 2')
    plt.title('Distance between structures')
    plt.show()
    
# %%
