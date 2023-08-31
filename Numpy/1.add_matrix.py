import numpy as np

mat1 = np.random.randint(100,size=25).reshape(5,5)
mat2 = np.arange(25).reshape(5,5)

sum = mat1+mat2

print(f"The Matrix 1:\n{mat1}\n")

print(f"The Matrix 2:\n{mat2}\n")

print(f"The Sum of Matrix 1 & 2:\n{sum}\n")