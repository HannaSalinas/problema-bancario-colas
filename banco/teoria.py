"""Resultados teóricos para comparar con la simulación.

Fórmulas M/M/1 de la Semana 6 (con μ = 1 / E[S]):
    ρ = λ/μ,  L = ρ/(1-ρ),  Lq = ρ²/(1-ρ),  W = 1/(μ-λ),  Wq = ρ/(μ-λ)

En este problema el tiempo de servicio de una caja es una mezcla de
exponenciales (cada clase tiene su propia media), así que no es exponencial y
la fórmula M/M/1 es solo una aproximación. Por eso también se calcula la
fórmula de Pollaczek-Khinchine (M/G/1), que usa el segundo momento:
    Wq = λ·E[S²] / (2(1-ρ))
Las llegadas tampoco son exactamente de Poisson: el tiempo entre llegadas es
una mezcla de exponenciales. Ambas fórmulas son aproximaciones de estado
estable, y la jornada es finita y empieza con el banco vacío.

Para el escenario R (fila única) se usa la fórmula de Erlang C del M/M/c.
"""

import math

import numpy as np
import pandas as pd

from banco import parametros as par


def momentos_llegada():
    """Media y segundo momento del tiempo entre llegadas de la fila única."""
    media = float(np.sum(par.PROB_CLASE * par.MEDIA_LLEGADA))
    segundo = float(np.sum(par.PROB_CLASE * 2 * par.MEDIA_LLEGADA ** 2))
    return media, segundo


def tasa_llegada():
    """Tasa total de llegadas λ (clientes por minuto)."""
    return 1.0 / momentos_llegada()[0]


def carga_ocho_procesos(num_cajas=3):
    """Carga si se leyera la Tabla 1 como 8 procesos de llegada independientes.

    Cada clase llegaría con tasa 1/media_llegada sin importar el 70/30 ni la
    Tabla 2. Se usa para justificar por qué se descarta esa lectura.
    """
    tasa = float(np.sum(1.0 / par.MEDIA_LLEGADA))
    carga = float(np.sum(par.MEDIA_SERVICIO / par.MEDIA_LLEGADA))
    return {"lambda": tasa, "carga": carga, "rho": carga / num_cajas}


def coeficientes_variacion():
    """Cuadrado del coeficiente de variación de llegadas y servicio.

    Para una exponencial vale 1. Valores mayores indican más variabilidad,
    lo que aumenta la espera frente a lo que predice el modelo M/M/1.
    """
    media_a, segundo_a = momentos_llegada()
    caja = parametros_caja(par.ESCENARIOS[0])[0]
    return {"ca2": (segundo_a - media_a ** 2) / media_a ** 2,
            "cs2": (caja["es2"] - caja["es"] ** 2) / caja["es"] ** 2}


def parametros_caja(escenario):
    """Tasa de llegada y momentos del servicio de cada caja del escenario.

    Un cliente de la acción a va a cada una de las n_a cajas que atienden esa
    acción con probabilidad 1/n_a. La mezcla de servicio de la caja pondera
    cada clase por la tasa con que llega a esa caja.

    Returns:
        Lista de diccionarios con lambda, es (E[S]) y es2 (E[S²]) por caja.
    """
    tasa = tasa_llegada()
    salida = []
    for k in range(escenario.num_cajas):
        pesos = np.zeros_like(par.PROB_CLASE)
        for accion in escenario.cajas[k]:
            n_cajas = len(escenario.cajas_para(accion))
            pesos[accion] = tasa * par.PROB_CLASE[accion] / n_cajas
        lam = float(pesos.sum())
        proporcion = pesos / lam
        salida.append({
            "lambda": lam,
            "es": float(np.sum(proporcion * par.MEDIA_SERVICIO)),
            "es2": float(np.sum(proporcion * 2 * par.MEDIA_SERVICIO ** 2)),
        })
    return salida


def mm1(lam, media_servicio):
    """Métricas de estado estable del M/M/1 (Semana 6)."""
    mu = 1.0 / media_servicio
    rho = lam / mu
    if rho >= 1:
        return {"rho": rho, "lq": np.inf, "wq": np.inf, "w": np.inf,
                "l": np.inf}
    return {"rho": rho, "lq": rho ** 2 / (1 - rho), "wq": rho / (mu - lam),
            "w": 1 / (mu - lam), "l": rho / (1 - rho)}


def mg1(lam, media_servicio, segundo_momento):
    """Métricas del M/G/1 con la fórmula de Pollaczek-Khinchine."""
    rho = lam * media_servicio
    if rho >= 1:
        return {"rho": rho, "lq": np.inf, "wq": np.inf, "w": np.inf,
                "l": np.inf}
    espera = lam * segundo_momento / (2 * (1 - rho))
    return {"rho": rho, "lq": lam * espera, "wq": espera,
            "w": espera + media_servicio,
            "l": lam * (espera + media_servicio)}


def mmc(lam, media_servicio, servidores):
    """Métricas del M/M/c con la fórmula de Erlang C."""
    carga = lam * media_servicio
    rho = carga / servidores
    if rho >= 1:
        return {"rho": rho, "lq": np.inf, "wq": np.inf, "w": np.inf,
                "l": np.inf}
    suma = sum(carga ** n / math.factorial(n) for n in range(servidores))
    ultimo = carga ** servidores / (math.factorial(servidores) * (1 - rho))
    prob_esperar = ultimo / (suma + ultimo)
    lq = prob_esperar * rho / (1 - rho)
    return {"rho": rho, "lq": lq, "wq": lq / lam, "w": lq / lam + media_servicio,
            "l": lq + carga}


def tabla_teoria(escenarios=par.ESCENARIOS):
    """Tabla de valores teóricos por caja y escenario.

    Para E1 a E4 hay una fila por caja (M/M/1 y P-K) y una fila global con
    los promedios ponderados por la tasa de llegada de cada caja. Para R hay
    una sola fila global con Erlang C.
    """
    filas = []
    for escenario in escenarios:
        if escenario.fila_unica:
            lam = tasa_llegada()
            caja = parametros_caja(par.ESCENARIOS[0])[0]
            media = caja["es"]
            res = mmc(lam, media, escenario.num_cajas)
            filas.append({"escenario": escenario.codigo, "grupo": "Total",
                          "modelo": f"M/M/{escenario.num_cajas}",
                          "lambda": lam, "es": media, **res})
            continue
        cajas = parametros_caja(escenario)
        acumulado = {"mm1": [], "pk": []}
        for k, caja in enumerate(cajas):
            for modelo, res in (("mm1", mm1(caja["lambda"], caja["es"])),
                                ("pk", mg1(caja["lambda"], caja["es"],
                                           caja["es2"]))):
                acumulado[modelo].append((caja["lambda"], res))
                filas.append({"escenario": escenario.codigo,
                              "grupo": escenario.etiqueta_caja(k),
                              "modelo": "M/M/1" if modelo == "mm1" else "P-K",
                              "lambda": caja["lambda"], "es": caja["es"],
                              **res})
        for modelo, lista in acumulado.items():
            lam_total = sum(lam for lam, _ in lista)
            wq = sum(lam * r["wq"] for lam, r in lista) / lam_total
            w = sum(lam * r["w"] for lam, r in lista) / lam_total
            filas.append({"escenario": escenario.codigo, "grupo": "Total",
                          "modelo": "M/M/1" if modelo == "mm1" else "P-K",
                          "lambda": lam_total,
                          "es": w - wq,
                          "rho": float(np.mean([r["rho"] for _, r in lista])),
                          "lq": sum(r["lq"] for _, r in lista), "wq": wq,
                          "w": w, "l": sum(r["l"] for _, r in lista)})
    return pd.DataFrame(filas)
