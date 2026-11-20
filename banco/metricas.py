"""Métricas por réplica y resúmenes estadísticos entre réplicas.

Niveles de agregación de cada réplica:
  * caja: una fila por caja del escenario;
  * clase: las 8 combinaciones acción-tipo (por ejemplo, Retiro-Rápido);
  * tipo: los 4 tipos de usuario sin separar la acción;
  * accion: retiro y pago;
  * global: todo el escenario.

La longitud promedio de fila se calcula sobre la duración real de la jornada
(480 minutos más el tiempo extra). Como el banco empieza y termina vacío,
la integral de la longitud de fila es igual a la suma de las esperas, y por
eso se cumple Lq = λ·Wq de forma exacta con λ = atendidos / duración.
Del mismo modo, L = Lq + (tiempo ocupado / duración).
"""

import numpy as np
import pandas as pd
from scipy import stats

from banco import parametros as par

METRICAS = ("atendidos", "t_atencion", "espera", "sistema", "lq", "l",
            "utilizacion", "espera_max")

NOMBRES_METRICAS = {
    "atendidos": "Clientes atendidos",
    "t_atencion": "Tiempo promedio de atención (min)",
    "espera": "Espera promedio en fila Wq (min)",
    "sistema": "Tiempo promedio en el sistema W (min)",
    "lq": "Longitud promedio de la fila Lq (clientes)",
    "l": "Clientes promedio en el sistema L",
    "utilizacion": "Utilización ρ",
    "espera_max": "Espera máxima (min)",
    "al_cierre": "Clientes en el banco al cierre",
    "tiempo_extra": "Tiempo extra después del cierre (min)",
}


def _metricas_grupo(mascara, resultado):
    """Métricas de un subconjunto de clientes de un día simulado."""
    espera = resultado.espera[mascara]
    num = int(mascara.sum())
    if num == 0:
        return {"atendidos": 0, "t_atencion": np.nan, "espera": np.nan,
                "sistema": np.nan, "lq": 0.0, "espera_max": np.nan}
    return {
        "atendidos": num,
        "t_atencion": float(resultado.clientes.servicio[mascara].mean()),
        "espera": float(espera.mean()),
        "sistema": float(resultado.sistema[mascara].mean()),
        "lq": float(espera.sum() / resultado.duracion),
        "espera_max": float(espera.max()),
    }


def filas_metricas(resultado, replica):
    """Construye las filas de métricas de un día simulado.

    Args:
        resultado: ResultadoDia de simulador.simular.
        replica: número de la réplica (empieza en 1).

    Returns:
        Lista de diccionarios, uno por grupo y nivel.
    """
    escenario = resultado.escenario
    clientes = resultado.clientes
    base = {"replica": replica, "escenario": escenario.codigo}
    filas = []

    for k in range(escenario.num_cajas):
        fila = {**base, "nivel": "caja", "grupo": escenario.etiqueta_caja(k),
                **_metricas_grupo(resultado.caja == k, resultado)}
        fila["utilizacion"] = resultado.tiempo_ocupado[k] / resultado.duracion
        if escenario.fila_unica:
            # Con fila compartida no se puede atribuir la fila a una caja.
            fila["lq"] = np.nan
        # L = Lq + clientes en servicio promedio (= utilización de la caja).
        fila["l"] = fila["lq"] + fila["utilizacion"]
        filas.append(fila)

    for accion, nombre_accion in enumerate(par.NOMBRES_ACCION):
        for tipo, nombre_tipo in enumerate(par.NOMBRES_TIPO):
            mascara = (clientes.accion == accion) & (clientes.tipo == tipo)
            filas.append({**base, "nivel": "clase",
                          "grupo": f"{nombre_accion}-{nombre_tipo}",
                          **_metricas_grupo(mascara, resultado)})

    for tipo, nombre_tipo in enumerate(par.NOMBRES_TIPO):
        filas.append({**base, "nivel": "tipo", "grupo": nombre_tipo,
                      **_metricas_grupo(clientes.tipo == tipo, resultado)})

    for accion, nombre_accion in enumerate(par.NOMBRES_ACCION):
        filas.append({**base, "nivel": "accion", "grupo": nombre_accion,
                      **_metricas_grupo(clientes.accion == accion,
                                        resultado)})

    total = {**base, "nivel": "global", "grupo": "Total",
             **_metricas_grupo(np.ones(len(clientes), dtype=bool), resultado)}
    total["lq"] = float(resultado.area_fila.sum() / resultado.duracion)
    total["utilizacion"] = float(resultado.tiempo_ocupado.mean()
                                 / resultado.duracion)
    total["l"] = total["lq"] + float(resultado.tiempo_ocupado.sum()
                                     / resultado.duracion)
    total["al_cierre"] = resultado.clientes_al_cierre
    total["tiempo_extra"] = resultado.tiempo_extra
    filas.append(total)
    return filas


def intervalo_t(valores, confianza=par.CONFIANZA):
    """Media, desviación estándar e IC con t de Student entre réplicas.

    Returns:
        Diccionario con n, media, de, me (margen de error), ic_inf e ic_sup.
    """
    datos = np.asarray(valores, dtype=float)
    datos = datos[~np.isnan(datos)]
    num = len(datos)
    if num == 0:
        return {"n": 0, "media": np.nan, "de": np.nan, "me": np.nan,
                "ic_inf": np.nan, "ic_sup": np.nan}
    media = float(datos.mean())
    if num < 2:
        return {"n": num, "media": media, "de": np.nan, "me": np.nan,
                "ic_inf": np.nan, "ic_sup": np.nan}
    desv = float(datos.std(ddof=1))
    t_critico = stats.t.ppf(1 - (1 - confianza) / 2, num - 1)
    margen = float(t_critico * desv / np.sqrt(num))
    return {"n": num, "media": media, "de": desv, "me": margen,
            "ic_inf": media - margen, "ic_sup": media + margen}


def resumir(datos, confianza=par.CONFIANZA):
    """Resume las métricas por escenario, nivel y grupo.

    Args:
        datos: DataFrame con una fila por réplica, escenario, nivel y grupo.
        confianza: nivel de confianza del intervalo.

    Returns:
        DataFrame largo con columnas escenario, nivel, grupo, metrica, n,
        media, de, me, ic_inf, ic_sup.
    """
    metricas = [m for m in (*METRICAS, "al_cierre", "tiempo_extra")
                if m in datos.columns]
    filas = []
    for (escenario, nivel, grupo), grupo_df in datos.groupby(
            ["escenario", "nivel", "grupo"], sort=False):
        for metrica in metricas:
            valores = grupo_df[metrica].to_numpy(dtype=float)
            if np.all(np.isnan(valores)):
                continue
            filas.append({"escenario": escenario, "nivel": nivel,
                          "grupo": grupo, "metrica": metrica,
                          **intervalo_t(valores, confianza)})
    return pd.DataFrame(filas)
