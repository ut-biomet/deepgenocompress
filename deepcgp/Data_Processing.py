# ==============================
# Tanzila Islam
# Email: tanzilamohita@gmail.com
# ===============================

import numpy as np

def one_hot_encode_snp_array(snp_array):
    encoding_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    onehotlabels = []
    for row in snp_array:
        encoded_row = []
        for base in row:
            encoded_row.extend(encoding_map.get(base, [0, 0, 0, 0]))
        onehotlabels.append(encoded_row)
    return np.array(onehotlabels, dtype=np.uint8)
