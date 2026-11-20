"""Validación vectorizada con la recursión de Lindley.

En una cola de un servidor con disciplina FIFO, la espera del cliente j+1 es

    Wq[j+1] = max(0, Wq[j] + S[j] - (A[j+1] - A[j]))

donde A son los instantes de llegada y S los tiempos de servicio. La recursión
solo depende de las entradas, así que se puede calcular para muchas réplicas
a la vez: cada réplica es una fila de una matriz y el ciclo recorre las
columnas (los clientes) con operaciones de NumPy sobre todas las filas.

Solo aplica a los escenarios con una fila por caja (E1 a E4). El escenario R
tiene una fila compartida por varias cajas y no cumple esta recursión.
"""

from dataclasses import dataclass

import numpy as np

from banco import parametros as par
from banco.clientes import asignar_cajas


@dataclass
class ResultadoLindley:
    """Métricas por réplica calculadas con Lindley.

    Cada arreglo tiene una fila por réplica; los que van por caja tienen una
    columna por caja.
    """

    atendidos: np.ndarray
    t_atencion: np.ndarray
    espera: np.ndarray
    espera_max: np.ndarray
    utilizacion: np.ndarray
    lq: np.ndarray
    espera_global: np.ndarray
    duracion: np.ndarray
    esperas_por_caja: list


def _matriz_rellena(lista_arreglos, valor_relleno):
    """Apila arreglos de distinto largo en una matriz rellena."""
    largo = max((len(a) for a in lista_arreglos), default=0)
    matriz = np.full((len(lista_arreglos), largo), valor_relleno, dtype=float)
    for fila, arreglo in enumerate(lista_arreglos):
        matriz[fila, :len(arreglo)] = arreglo
    return matriz


def esperas_lindley(llegadas, servicios):
    """Aplica la recursión de Lindley a una matriz de réplicas.

    Args:
        llegadas: matriz (réplicas x clientes) con los instantes de llegada
            en orden; las posiciones vacías tienen NaN.
        servicios: matriz del mismo tamaño con los tiempos de servicio.

    Returns:
        Matriz de esperas en fila; NaN donde no hay cliente.
    """
    espera = np.full(llegadas.shape, np.nan)
    if llegadas.shape[1] == 0:
        return espera
    espera[:, 0] = np.where(np.isnan(llegadas[:, 0]), np.nan, 0.0)
    for j in range(1, llegadas.shape[1]):
        entre = llegadas[:, j] - llegadas[:, j - 1]
        espera[:, j] = np.maximum(0.0,
                                  espera[:, j - 1] + servicios[:, j - 1]
                                  - entre)
    return espera


def _compactar(mascara, llegada, servicio):
    """Junta a la izquierda, fila por fila, los valores donde la máscara es 1.

    Sirve para pasar de la matriz con todos los clientes de cada réplica a la
    matriz con solo los clientes de una caja, sin ciclos sobre las réplicas.
    """
    posicion = np.cumsum(mascara, axis=1) - 1
    largo = int(mascara.sum(axis=1).max(initial=0))
    filas, columnas = np.nonzero(mascara)
    salida = []
    for matriz in (llegada, servicio):
        compacta = np.full((mascara.shape[0], largo), np.nan)
        compacta[filas, posicion[filas, columnas]] = matriz[filas, columnas]
        salida.append(compacta)
    return salida[0], salida[1]


def _maximo_por_fila(matriz):
    """Máximo de cada fila ignorando NaN (0 si la fila está vacía)."""
    return np.max(np.where(np.isnan(matriz), 0.0, matriz), axis=1,
                  initial=0.0)


def simular_lindley(lista_clientes, escenario, horizonte=par.HORIZONTE):
    """Calcula las métricas por caja de muchas réplicas a la vez.

    Args:
        lista_clientes: lista con un objeto Clientes por réplica.
        escenario: escenario con una fila por caja (no fila única).
        horizonte: minuto de cierre de puertas.

    Returns:
        ResultadoLindley con las métricas de cada réplica.
    """
    if escenario.fila_unica:
        raise ValueError("Lindley solo aplica a cajas con fila propia")
    # Matrices con todos los clientes de cada réplica (relleno: NaN / -1).
    llegada = _matriz_rellena([c.llegada for c in lista_clientes], np.nan)
    servicio = _matriz_rellena([c.servicio for c in lista_clientes], np.nan)
    caja = _matriz_rellena([asignar_cajas(c, escenario)
                            for c in lista_clientes], -1).astype(np.int64)

    columnas = {nombre: [] for nombre in
                ("atendidos", "ocupado", "suma_espera", "espera_max")}
    esperas_por_caja = []
    ultima_salida = np.zeros(len(lista_clientes))
    for k in range(escenario.num_cajas):
        llegada_k, servicio_k = _compactar(caja == k, llegada, servicio)
        espera_k = esperas_lindley(llegada_k, servicio_k)
        esperas_por_caja.append(espera_k)
        columnas["atendidos"].append(np.sum(~np.isnan(llegada_k), axis=1))
        columnas["ocupado"].append(np.nansum(servicio_k, axis=1))
        columnas["suma_espera"].append(np.nansum(espera_k, axis=1))
        columnas["espera_max"].append(_maximo_por_fila(espera_k))
        ultima_salida = np.maximum(
            ultima_salida, _maximo_por_fila(llegada_k + espera_k + servicio_k))

    tabla = {nombre: np.column_stack(valores)
             for nombre, valores in columnas.items()}
    duracion = np.maximum(horizonte, ultima_salida)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ResultadoLindley(
            atendidos=tabla["atendidos"],
            t_atencion=tabla["ocupado"] / tabla["atendidos"],
            espera=tabla["suma_espera"] / tabla["atendidos"],
            espera_max=tabla["espera_max"],
            utilizacion=tabla["ocupado"] / duracion[:, None],
            lq=tabla["suma_espera"] / duracion[:, None],
            espera_global=(tabla["suma_espera"].sum(axis=1)
                           / tabla["atendidos"].sum(axis=1)),
            duracion=duracion,
            esperas_por_caja=esperas_por_caja,
        )
