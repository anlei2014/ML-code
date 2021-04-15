import numpy as np

rdm = np.random.RandomState(seed=1)
a = rdm.rand()
b = rdm.rand(3, 3)
d = rdm.rand()
c = rdm.rand(2, 3)
print("a:", a)
print("b:", b)
print("c:", c)
print("d:", d)
