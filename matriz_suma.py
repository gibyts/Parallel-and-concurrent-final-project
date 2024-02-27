import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def add_matrices_sequential(matrixA, matrixB):
    if matrixA.shape != matrixB.shape:
        raise ValueError("Las matrices deben tener las mismas dimensiones para poder sumarlas.")
    
    result = matrixA + matrixB
    return result

def add_matrices_parallel(matrixA, matrixB):
    if matrixA.shape != matrixB.shape:
        raise ValueError("Las matrices deben tener las mismas dimensiones para poder sumarlas.")

    def add_portion(start, end):
        nonlocal result
        result[start:end, :] = matrixA[start:end, :] + matrixB[start:end, :]
    
    filas, columnas = matrixA.shape
    num_threads = min(filas, multiprocessing.cpu_count())

    result = np.zeros(matrixA.shape)
    chunk_size = matrixA.shape[0] // num_threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i in range(0, matrixA.shape[0], chunk_size):
            end_idx = min(i + chunk_size, matrixA.shape[0])
            futures.append(executor.submit(add_portion, i, end_idx))

        for future in futures:
            future.result()

    return result, num_threads

def compare_results(sequential_result, parallel_result, sequential_time, parallel_time, num_threads):
    print("\n\t\t\tResults:\n")
    print(f"Sequential matrix multiplication time: {sequential_time:.4f} seconds")
    print(f"  Parallel matrix multiplication time: {parallel_time:.4f} seconds")
    if parallel_time == 0:
        print("Parallel matrix addition time is zero, cannot calculate speedup and efficiency.")
    else:
        speedup = sequential_time / parallel_time
        print(f"\n   Speedup: {speedup:.4f}")
        efficiency = (speedup / num_threads) * 100
        print(f"Efficiency: {efficiency:.4f}%")