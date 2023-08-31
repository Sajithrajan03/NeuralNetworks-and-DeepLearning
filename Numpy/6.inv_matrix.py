import numpy as np

mat1 = np.arange(1,5).reshape(2,2)


inv = np.linalg.inv(mat1)

print(f"The Matrix 1:\n{mat1}\n")
print(f"The dim : {mat1.shape}\n")
det = np.linalg.det(mat1)

 
print(f"\nThe Determinant : {det}\n")

print(f"\nThe Inverse Matrix 1 :\n{inv}\n")
print(f"The dim : {inv.shape}\n")