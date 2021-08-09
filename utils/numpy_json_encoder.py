import json

import numpy as np


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            if o.ndim == 1:
                return list(o)
            elif o.ndim == 2:
                return list(map(list, o))
            else:
                raise ValueError(
                    f"Cannot serialize numpy array with ndim: {o.ndim}")
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.integer):
            return int(o)

        return super().default(o)
