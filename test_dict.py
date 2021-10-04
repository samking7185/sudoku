"""
test.py
~~~~~~~~~~~~
This shows the structure and how to call a FIS using fuzzy_struc.py
"""

from fuzzy_struc_dict import FIS
import time

rules = {
    1: {"mf": ["Low", "Low", "Low"], "oper": "OR"},
    2: {"mf": ["Low", "Medium", "High"], "oper": "AND"},
    3: {"mf": ["Low", "High", "High"], "oper": "AND"},
    4: {"mf": ["High", "Low", "Low"], "oper": "AND"},
    5: {"mf": ["High", "Medium", "High"], "oper": "AND"},
    6: {"mf": ["High", "High", "High"], "oper": "AND"}
}

MF1 = {
    "Low": {"value": (0, 0, 3), "shape": "triangle"},
    "High": {"value": (1, 4, 4), "shape": "triangle"}
       }

MF2 = {
    "Low": {"value": (0, 0, 2), "shape": "triangle"},
    "Medium": {"value": (1, 2, 3), "shape": "triangle"},
    "High": {"value": (2, 4, 4), "shape": "triangle"}
       }

MFo = {
    "Low": {"value": (0, 0, 1), "shape": "triangle"},
    "High": {"value": (0, 1, 1), "shape": "triangle"}
       }

parameters = {
    "MF1": MF1,
    "MF2": MF2,
    "OUT": MFo,
    "RULES": rules,
    "BOUNDS": [0, 1]
}

tic = time.perf_counter()
single = FIS(parameters)

out1 = single.compute(3.5, 1)
toc = time.perf_counter()

print(out1)
print(toc - tic)
