import numpy as np
import multiprocessing

def gauss_jordan_inverse(matrix):
    n = len(matrix)
    augmented_matrix = np.concatenate((matrix, np.identity(n)), axis=1)
    
    for i in range(n):
        augmented_matrix[i] /= augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]
    
    inverse_matrix = augmented_matrix[:, n:]
    
    return inverse_matrix

def calculate_inverse_sequential(matrix):
    inverse = gauss_jordan_inverse(matrix)
    return inverse

def calculate_inverse_parallel(matrix):
    filas, columnas = matrix.shape
    num_threads = min(filas, multiprocessing.cpu_count())
    pool = multiprocessing.Pool(processes=num_threads)
    inverse = pool.apply(gauss_jordan_inverse, (matrix,))
    pool.close()
    pool.join()
    return inverse,num_threads
