import numpy as np

mat1 = np.arange(2,11).reshape(3,3)


det = np.linalg.det(mat1)

print(f"The Matrix 1:\n{mat1}\n")
print(f"The Determinant : {det}\n")


