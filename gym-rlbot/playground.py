import numpy as np
import math
from decimal import *

# def convolveArgMax(y, f):
#     np.seterr(divide='ignore', invalid='ignore')
#     ptr = 0
#     ptr_end = ptr + len(f)
#
#     result = []
#     while (ptr_end <= len(y)):
#         out = np.corrcoef([y[ptr:ptr_end], f])
#         if math.isnan(out[0, 1]):
#             result.append(0)
#         else:
#             result.append(out[0, 1])
#         ptr += 1
#         ptr_end = ptr + len(f)
#
#     return np.argmax(result)
#
# y = np.array([0, 0, 0, 0, 0, 2, 3, 4, 3, 3, 3, 3, 3, 3, 3])
# f = np.array([Decimal(2), 3, 4])
#
# f = map(lambda x: float(x), f)
#
# print (convolveArgMax(y, f))
a = np.empty((5, 5, 5))
b = [a, a, a]
c = np.concatenate(b , axis=2)
print (c.shape)
