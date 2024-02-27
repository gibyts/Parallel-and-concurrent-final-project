import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing


def calcular_transpuesta_secuencial(matriz):
    filas, columnas = len(matriz), len(matriz[0])
    nueva_matriz = [[0] * filas for _ in range(columnas)]
    
    for i in range(filas):
        for j in range(columnas):
            nueva_matriz[j][i] = matriz[i][j]

    nueva_matriz = np.array(nueva_matriz)
    return nueva_matriz

def calcular_traspuesta_paralela(matriz):
    filas, columnas = matriz.shape

    num_hilos = min(filas, multiprocessing.cpu_count())
    with ThreadPoolExecutor(max_workers=num_hilos) as executor:
        # Calcular el rango de filas para cada hilo
        rango_filas = [(i * filas // num_hilos, (i + 1) * filas // num_hilos) for i in range(num_hilos)]

        # Transponer la matriz parcial en paralelo
        resultados = list(executor.submit(transponer_parcial, matriz, rango).result() for rango in rango_filas)

    # Combinar resultados parciales en una matriz completa
    traspuesta_completa = np.concatenate(resultados, axis=1)

    return traspuesta_completa, num_hilos

def transponer_parcial(matriz, rango_filas):
    inicio, fin = rango_filas
    return np.transpose(matriz[inicio:fin, :])

