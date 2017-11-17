import numpy as np
import time
import os

N = 8000

arr = []
A = np.random.random((N, N)).astype(np.float32)

for i in range(5):
    print('Test %d...' % i)
    st = time.time()
    B = np.dot(A, A)
    B[0,0]
    ed = time.time()
    dif = ed - st
    arr.append(dif)

avg = np.mean(arr)


print('avg:{:.2f}s'.format(avg))
