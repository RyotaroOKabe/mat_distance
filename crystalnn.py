#%%
from mp_utils import *
cnn = CrystalNN()
structure1, structure2 = [get_mpstruct(mpid) for mpid in ['mp-2534', 'mp-20536']]
# Analyze structure 1
local_env_structure1 = [cnn.get_nn_info(structure1, i) for i in range(len(structure1))]
# Analyze structure 2
local_env_structure2 = [cnn.get_nn_info(structure2, i) for i in range(len(structure2))]

import numpy as np
from scipy.spatial.distance import euclidean
# Simplified example: Create feature vectors based on average coordination numbers
avg_cn_structure1 = np.mean([len(env) for env in local_env_structure1])
avg_cn_structure2 = np.mean([len(env) for env in local_env_structure2])
# Compare using Euclidean distance (lower values indicate more similarity)
similarity_score = euclidean([avg_cn_structure1], [avg_cn_structure2])
print(f"Similarity score (lower is more similar): {similarity_score}")

    
# %%
