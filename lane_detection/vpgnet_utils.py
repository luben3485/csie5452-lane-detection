import numpy as np

def resize_array(array_in, shape=None):
    if shape is None:
        return array_in
    m, n = shape
    k1, k2 = array_in.shape

    if m == k1 and n == k2:
        return array_in
    else:
        y = np.zeros((m, n), dtype=type(array_in[0, 0]))

        # k = len(array_in)
        p, q = k1 / m, k2 / n
        for i in range(m):
            y[i, :] = array_in[np.int_(i * p), np.int_(np.arange(n) * q)]
        return y
