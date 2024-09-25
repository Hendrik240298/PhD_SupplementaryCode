import numpy as np

SIZE = 2696
matrix = np.zeros(shape=(SIZE, SIZE))

with open("pressure_laplace_matrix.txt") as f:
    for line in f:
        a, b = line.split(" ")
        b = b.strip("\n")
        a = a.strip("(").strip(")")

        i, j = a.split(",")
        i = int(i)
        j = int(j)
        val = np.float(b)

        matrix[i][j] = val

print("System matrix is symmetric:           ", (matrix.transpose() == matrix).all())
print("System matrix has rank:               ", np.linalg.matrix_rank(matrix))
print("System matrix has condition number:   ", round(np.linalg.cond(matrix), 2))

matrix_inverse = np.linalg.inv(matrix)
print(matrix_inverse.dot(matrix))
