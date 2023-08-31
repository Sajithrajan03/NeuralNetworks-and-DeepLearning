import numpy as np

mat1 = np.arange(24).reshape(6,4)


trans = mat1.T

print(f"The Matrix 1:\n{mat1}\n")
print(f"The dim : {mat1.shape}\n")


print(f"\nThe Transpose Matrix 1 :\n{trans}\n")
print(f"The dim : {trans.shape}\n")