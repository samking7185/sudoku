"""
test.py
~~~~~~~~~~~~
This shows the structure and how to call a FIS using fuzzy_struc.py
"""

from fuzzy_struc import FIS
import time

parameters = {
    "MF1": [(0, 0, 3), (1, 4, 4)],
    "MF2": [(0, 0, 2), (1, 2, 3), (2, 4, 4)],
    "OUT": [(0, 0, 1), (0, 1, 1)],
    "RULES": [0, 1, 1, 0, 1, 1],
    "BOUNDS": [0, 1]
}

tic = time.perf_counter()
single = FIS(parameters)

out1 = single.compute(3, 2)
toc = time.perf_counter()

print(out1)
print(toc - tic)
