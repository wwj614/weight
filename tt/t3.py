import numpy as np

vals = np.loadtxt('./c.txt', usecols=0, delimiter='\t', unpack=True)
steps = (0.125, 0.25, 0.5, 1, 2, 4)
result = []
for step in steps:
    i1 = list(map((lambda x: int(x / step)), vals))
    i2 = list(zip(i1, i1[1:]))
    i3 = list(zip(i1, i1[1:], i1[2:]))
    i4 = list(zip(i1, i1[1:], i1[2:], i1[3:]))
    result.append((step, len(set(i2)), len(set(i3)), len(set(i4))))
print(result)
