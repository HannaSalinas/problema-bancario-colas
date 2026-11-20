"""Cálculo del número de réplicas necesario (Semana 7).

Con n0 réplicas iniciales se calcula el error relativo de la media:
    ER0 = t(α/2, n0-1)·S / (√n0 · X̄)
y el número de réplicas para llegar al error objetivo ER*:
    n = n0 · (ER0 / ER*)²
Como t depende de n, también se calcula el método t iterativo: el menor n
con t(α/2, n-1)·S / (√n · X̄) ≤ ER*, usando la S de las réplicas iniciales.
"""

import math

import numpy as np
import pandas as pd
from scipy import stats

from banco import parametros as par
from banco.metricas import intervalo_t


N_MAXIMO = 1_000_000


def n_t_iterativo(media, desv, *, er_objetivo=par.ER_OBJETIVO,
                  confianza=par.CONFIANZA):
    """Menor n que cumple el error relativo con el valor t de n-1 grados."""
    for num in range(2, N_MAXIMO + 1):
        t_critico = stats.t.ppf(1 - (1 - confianza) / 2, num - 1)
        if t_critico * desv / (math.sqrt(num) * media) <= er_objetivo:
            return num
    return N_MAXIMO


def replicas_por_formula(num, media, desv, *, er_objetivo=par.ER_OBJETIVO,
                         confianza=par.CONFIANZA):
    """Error relativo con n réplicas y réplicas necesarias por la fórmula.

    Returns:
        Tupla (ER, n requerido por n0·(ER0/ER*)²).
    """
    t_critico = stats.t.ppf(1 - (1 - confianza) / 2, num - 1)
    error = t_critico * desv / (math.sqrt(num) * media)
    return error, math.ceil(num * (error / er_objetivo) ** 2)


def tabla_replicas(datos, escenarios=par.ESCENARIOS_CRITERIO, *,
                   er_objetivo=par.ER_OBJETIVO, confianza=par.CONFIANZA):
    """Tabla del cálculo de réplicas por escenario.

    El error relativo se calcula sobre la espera promedio global (Wq) de
    cada escenario.

    Args:
        datos: DataFrame de métricas por réplica (salida del experimento).
        escenarios: códigos de escenario que entran en el criterio.
        er_objetivo: error relativo objetivo.
        confianza: nivel de confianza.

    Returns:
        DataFrame con una fila por escenario: n0, media, S, t, ME, ER0,
        n por fórmula, n por t iterativo y n requerido.
    """
    globales = datos[datos["nivel"] == "global"]
    filas = []
    for codigo in escenarios:
        valores = globales.loc[globales["escenario"] == codigo, "espera"]
        resumen = intervalo_t(valores, confianza)
        num = resumen["n"]
        error, n_formula = replicas_por_formula(num, resumen["media"],
                                                resumen["de"],
                                                er_objetivo=er_objetivo,
                                                confianza=confianza)
        n_iter = n_t_iterativo(resumen["media"], resumen["de"],
                               er_objetivo=er_objetivo, confianza=confianza)
        filas.append({
            "escenario": codigo,
            "n0": num,
            "media": resumen["media"],
            "de": resumen["de"],
            "t": float(stats.t.ppf(1 - (1 - confianza) / 2, num - 1)),
            "me": resumen["me"],
            "er": error,
            "cumple": bool(error <= er_objetivo),
            "n_formula": n_formula,
            "n_t_iterativo": n_iter,
            "n_requerido": max(par.N_INICIAL, n_formula),
        })
    return pd.DataFrame(filas)


def n_final(tabla):
    """Número de réplicas final: el máximo requerido entre escenarios."""
    return int(np.max(tabla["n_requerido"]))
