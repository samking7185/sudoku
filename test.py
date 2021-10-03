from fuzzy_struc import FIS
import time

MF1vals = [(0, 0, 2), (1, 2, 4, 4)]
MF2vals = [(0, 0, 2), (1, 2, 4, 4)]
outputMF = [(0, 0, 1), (0, 1, 1)]
rules = [0, 1, 0, 1]

parameters = {
    "MF1": [(0, 0, 2), (1, 2, 4, 4)],
    "MF2": [(0, 0, 2), (1, 2, 4, 4)],
    "OUT": [(0, 0, 1), (0, 1, 1)],
    "RULES": [0, 1, 0, 1],
    "BOUNDS": [0, 1]
}

tic = time.perf_counter()
single = FIS(parameters)

out1 = single.compute(1.2, 1)
toc = time.perf_counter()

print(out1)
print(toc - tic)
