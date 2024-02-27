import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def multiply_matrices_sequential(matrixA, matrixB):
    result = np.dot(matrixA, matrixB)
    return result

def multiply_matrices_parallel(matrixA, matrixB):

    def multiply_portion(start, end):
        nonlocal result
        result[start:end, :] = np.dot(matrixA[start:end, :], matrixB)

    result = np.zeros((matrixA.shape[0], matrixB.shape[1]))
    filas, columnas = matrixA.shape
    num_threads = min(filas, multiprocessing.cpu_count())
    chunk_size = matrixA.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, matrixA.shape[0], chunk_size):
            end_idx = min(i + chunk_size, matrixA.shape[0])
            futures.append(executor.submit(multiply_portion, i, end_idx))

        for future in futures:
            future.result()

    return result, num_threads
