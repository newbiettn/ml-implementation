####################################################################################################
# vectorization_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          31 Mar 2021
# DESCRIPTION:      Benchmark vectorized vs regular computations.
####################################################################################################
import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

# Vectorized
tic = time.time()
inner_prod_1 = np.dot(a, b)
toc = time.time()
print(f'Vectorized version: c = %0.3f, duration = %0.3f ms' % (inner_prod_1, 1000*(toc-tic)))

# Non vectorized
inner_prod_2 = 0
tic = time.time()
for i in range(1000000):
     inner_prod_2 += a[i]*b[i]
toc = time.time()
print(f'Vectorized version: c = %0.3f, duration = %0.3f ms' % (inner_prod_2, 1000*(toc-tic)))


