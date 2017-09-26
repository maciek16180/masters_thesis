import numpy as np
from itertools import chain


def load_subtle_fixed(path, trim=200):
    data = np.load(path + 'subtle4reddit.pkl')

    if trim is not None:
        data_trimmed = []
        for q, a in data:
            if max(len(q), len(a)) <= trim:
                data_trimmed.append([q, a])
        data = data_trimmed

    return data
