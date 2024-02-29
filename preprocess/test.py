import torch
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LinearRegression

a = torch.tensor((1, 2, 3), dtype=torch.int)
b = torch.tensor((7, 2, 3), dtype=torch.int)
print(b/a)

a = torch.tensor((1, 2, 3), dtype=torch.int)
la = []
la.append(a)
a = torch.tensor((6, 2, 3), dtype=torch.int)
la.append(a)
la = torch.stack(la, dim=0)
print(la)

a = torch.tensor((1, 2, 3), dtype=torch.int)
la = torch.zeros((2, 3))
la[0, :] = a
a = torch.tensor((6, 2, 3), dtype=torch.int)
la[1, :] = a
print(la)

# Create a sparse COO matrix using scipy
matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.int32)
print(np.sum(matrix**2, axis=1))
a = matrix[0, 0]
b = matrix[0][0]
#matrix[:, 0] = 3
print(a == b)
print(matrix[0, 0])
print(matrix[0][0])
for i in range(0, 4):
    for j in range(3, 6):
        if j >= i:
            break
print(j)


dates = np.array([0, 2, 4, 6, 7, 8], dtype=int)
a = np.array([3.0, 0.0, 2.0], dtype=bool)
print(dates[0:2])
print(matrix[a, :])

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y_1 = np.dot(X, np.array([1, 2])) + 3
y_2 = np.dot(X, np.array([-6, 3])) -1
y = np.stack([y_1, y_2], axis=1)
print(y.shape)
reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.intercept_)
lr=1e-3
print(str(lr))
#coo_matrix = sp.coo_matrix([[1, 0, 2], [0, 3, 0]])
#edge_index = torch.tensor([coo_matrix.row, coo_matrix.col], dtype=torch.long)
#print(edge_index)
# Convert the sparse COO matrix to a torch sparse COO tensor
# indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
# shape = torch.Size(coo_matrix.shape)

# sparse_tensor = torch.sparse_coo_tensor(indices)

# Print the sparse tensor
# print(sparse_tensor)

matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.int32)
a = np.array([1, 2, 3])
a = np.expand_dims(a, axis=-1)
b = np.concatenate((matrix, a), axis=-1)
c = np.array([False, True, False, True])
print(b[:, c])

a = np.array([1, 2, 3])
s = a
s[1] = 10
print(a)
print(s)
import copy
s = copy.copy(a)
s[1] = 100
print(a)
print(s)