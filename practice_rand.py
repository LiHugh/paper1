import numpy as np

size = 15
K = 5
for t in range(size):
    for k in range(K):
        future = np.random.randint(t, size)
        print('-------------', t)
        print(future)