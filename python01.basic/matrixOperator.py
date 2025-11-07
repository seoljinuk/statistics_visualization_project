import numpy as np

data_a = [1, 0, -2, 2, 1, -1]
data_b = [2, 1, 1, 0, -1, 1]

array_a = np.array(data_a).reshape(2, 3)
array_b = np.array(data_b).reshape(3, 2)

result = np.dot(array_a, array_b)
print(result.shape)
print(result)