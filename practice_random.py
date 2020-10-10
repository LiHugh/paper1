import random
import numpy as np


t = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
t0 = len(t)
print(t0)

a = np.random.randint(2, size=10)
print(a)
b = np.copy(a)
print(b)

size = 5
buffer = []
for i in range(10):
    buffer.append(np.random.randint(2, size=10))

if len(buffer) >= size:
    experience_buffer = buffer
    print(experience_buffer)
else:
    experience_buffer = buffer * size

print(np.copy(np.reshape(np.array(random.sample(experience_buffer, size)), [size, 4])))
print(len(np.copy(np.reshape(np.array(random.sample(experience_buffer, size)), [size, 4]))))