"""Ejecución de réplicas de todos los escenarios.

En cada réplica se genera una sola secuencia de clientes y se simula en todos
los escenarios (números aleatorios comunes). Así las diferencias entre
escenarios se deben a la configuración de cajas y no al azar de las llegadas.
"""

import pandas as pd

from banco import parametros as par
from banco.clientes import flujos_replicas, generar_clientes
from banco.metricas import filas_metricas
from banco.simulador import simular


def tabla_clientes(resultado, replica):
    """DataFrame con los datos individuales de cada cliente de un día."""
    clientes = resultado.clientes
    return pd.DataFrame({
        "replica": replica,
        "escenario": resultado.escenario.codigo,
        "cliente": range(1, len(clientes) + 1),
        "accion": [par.NOMBRES_ACCION[a] for a in clientes.accion],
        "tipo": [par.NOMBRES_TIPO[t] for t in clientes.tipo],
        "caja": resultado.caja + 1,
        "llegada": clientes.llegada,
        "inicio": resultado.inicio,
        "fin": resultado.fin,
        "servicio": clientes.servicio,
        "espera": resultado.espera,
        "sistema": resultado.sistema,
    })


def correr(num_replicas, escenarios=par.ESCENARIOS, semilla=par.SEMILLA,
           guardar_clientes=False):
    """Corre las réplicas y devuelve las métricas por réplica.

    Args:
        num_replicas: número de réplicas.
        escenarios: escenarios a simular.
        semilla: semilla raíz.
        guardar_clientes: si es verdadero, también devuelve los datos de
            cada cliente.

    Returns:
        Tupla (métricas, clientes). clientes es None si no se pidió.
    """
    filas = []
    tablas = []
    for replica, (rng_c, rng_a) in enumerate(
            flujos_replicas(num_replicas, semilla), start=1):
        clientes = generar_clientes(rng_c, rng_a)
        for escenario in escenarios:
            resultado = simular(clientes, escenario)
            filas.extend(filas_metricas(resultado, replica))
            if guardar_clientes:
                tablas.append(tabla_clientes(resultado, replica))
    metricas = pd.DataFrame(filas)
    detalle = pd.concat(tablas, ignore_index=True) if guardar_clientes else None
    return metricas, detalle


def generar_lote(num_replicas, semilla=par.SEMILLA):
    """Genera las secuencias de clientes de varias réplicas (para Lindley)."""
    return [generar_clientes(rng_c, rng_a)
            for rng_c, rng_a in flujos_replicas(num_replicas, semilla)]
