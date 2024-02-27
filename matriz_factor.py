import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def multiply_matrix_scalar_sequential(matrix, scalar):
    
    result = matrix * scalar
    return result

def multiply_matrix_scalar_parallel(matrix, scalar):

    def multiply_portion(start, end):
        nonlocal result
        result[start:end, :] = matrix[start:end, :] * scalar

    filas, columnas = matrix.shape
    num_threads = min(filas, multiprocessing.cpu_count())

    result = np.zeros(matrix.shape)
    chunk_size = matrix.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, matrix.shape[0], chunk_size):
            end_idx = min(i + chunk_size, matrix.shape[0])
            futures.append(executor.submit(multiply_portion, i, end_idx))

        for future in futures:
            future.result()

    return result,num_threads