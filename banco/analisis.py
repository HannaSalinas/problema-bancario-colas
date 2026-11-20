"""Análisis de resultados: respuestas a P1-P5, pruebas y comparación teórica.

Todas las comparaciones entre escenarios usan réplicas pareadas: en cada
réplica los escenarios atienden la misma secuencia de clientes (números
aleatorios comunes), así que se compara la diferencia réplica a réplica con
una prueba t pareada y su intervalo de confianza.
"""

import time

import numpy as np
import pandas as pd
from scipy import stats

from banco import parametros as par
from banco import teoria
from banco.experimento import generar_lote
from banco.lindley import simular_lindley
from banco.metricas import intervalo_t
from banco.simulador import simular


def _serie(datos, escenario, nivel="global", grupo="Total",
           metrica="espera"):
    """Valores por réplica de una métrica, ordenados por réplica."""
    filtro = ((datos["escenario"] == escenario) & (datos["nivel"] == nivel)
              & (datos["grupo"] == grupo))
    return datos.loc[filtro].sort_values("replica")[metrica].to_numpy(float)


# ----------------------------------------------------------------------- P1

def p1_tiempo_atencion(datos, codigo="E1"):
    """Tiempo promedio de atención por caja y ANOVA de un factor.

    Returns:
        Tupla (tabla por caja, diccionario con la caja menor, la mayor y el
        resultado del ANOVA).
    """
    escenario = par.escenario_por_codigo(codigo)
    grupos = [escenario.etiqueta_caja(k) for k in range(escenario.num_cajas)]
    muestras = [_serie(datos, codigo, "caja", g, "t_atencion") for g in grupos]
    tabla = pd.DataFrame([{"escenario": codigo, "caja": g, **intervalo_t(m)}
                          for g, m in zip(grupos, muestras)])
    estadistico, valor_p = stats.f_oneway(*muestras)
    menor = tabla.loc[tabla["media"].idxmin()]
    mayor = tabla.loc[tabla["media"].idxmax()]
    return tabla, {
        "escenario": codigo,
        "caja_menor": menor["caja"], "media_menor": menor["media"],
        "caja_mayor": mayor["caja"], "media_mayor": mayor["media"],
        "f": float(estadistico), "p": float(valor_p),
        "significativa": bool(valor_p < 1 - par.CONFIANZA),
    }


# ------------------------------------------------------------------- P2, P3

def usuarios_por_tipo_replica(datos, codigo="E1", nivel="tipo"):
    """Tabla réplica x tipo con el número de usuarios (P3).

    Con números aleatorios comunes los conteos son iguales en todos los
    escenarios, porque solo dependen de la secuencia de clientes.
    """
    filtro = (datos["escenario"] == codigo) & (datos["nivel"] == nivel)
    tabla = datos.loc[filtro].pivot(index="replica", columns="grupo",
                                    values="atendidos")
    if nivel == "tipo":
        tabla = tabla[list(par.NOMBRES_TIPO)]
    else:
        tabla = tabla[[f"{a}-{t}" for a in par.NOMBRES_ACCION
                       for t in par.NOMBRES_TIPO]]
    tabla["Total"] = tabla.sum(axis=1)
    return tabla.astype(int)


def replicas_con_menos_usuarios(tabla):
    """Para cada tipo, la réplica (o réplicas, si hay empate) con menos."""
    filas = []
    for columna in tabla.columns:
        minimo = tabla[columna].min()
        replicas = tabla.index[tabla[columna] == minimo].tolist()
        filas.append({"tipo": columna, "minimo": int(minimo),
                      "replicas": ", ".join(str(r) for r in replicas),
                      "media": float(tabla[columna].mean()),
                      "maximo": int(tabla[columna].max())})
    return pd.DataFrame(filas)


def p2_promedio_usuarios(datos, codigo="E1"):
    """Promedio de usuarios por tipo en el total de cajas y por caja (P2)."""
    filas = []
    for nivel in ("tipo", "clase"):
        tabla = usuarios_por_tipo_replica(datos, codigo, nivel)
        for columna in tabla.columns:
            if nivel == "clase" and columna == "Total":
                continue
            res = intervalo_t(tabla[columna])
            filas.append({"nivel": nivel, "tipo": columna, **res,
                          "por_caja": res["media"]
                          / par.escenario_por_codigo(codigo).num_cajas})
    return pd.DataFrame(filas)


# ------------------------------------------------------------- comparaciones

def prueba_pareada(datos, codigo_a, codigo_b, *, grupo="Total",
                   metrica="espera"):
    """Prueba t pareada de la diferencia A - B entre escenarios.

    grupo es "Total" (nivel global) o una acción ("Retiro" o "Pago").
    """
    nivel = "global" if grupo == "Total" else "accion"
    serie_a = _serie(datos, codigo_a, nivel, grupo, metrica)
    serie_b = _serie(datos, codigo_b, nivel, grupo, metrica)
    diferencia = serie_a - serie_b
    estadistico, valor_p = stats.ttest_rel(serie_a, serie_b)
    res = intervalo_t(diferencia)
    return {"comparacion": f"{codigo_a} - {codigo_b}", "metrica": metrica,
            "grupo": grupo, "media_a": float(serie_a.mean()),
            "media_b": float(serie_b.mean()), "diferencia": res["media"],
            "ic_inf": res["ic_inf"], "ic_sup": res["ic_sup"],
            "t": float(estadistico), "p": float(valor_p),
            "significativa": bool(valor_p < 1 - par.CONFIANZA),
            "replicas_a_menor": float(np.mean(diferencia < 0))}


def mejor_exclusiva(datos):
    """Código de la combinación exclusiva (E2 o E3) con menor Wq global."""
    medias = {c: _serie(datos, c).mean() for c in ("E2", "E3")}
    return min(medias, key=medias.get)


def pruebas_escenarios(datos):
    """Pruebas pareadas que sustentan P4, P5 y el efecto de la fila única."""
    exclusiva = mejor_exclusiva(datos)
    pares = [("E2", "E3"), ("E1", exclusiva), ("E4", "E1"), ("R", "E1")]
    filas = [prueba_pareada(datos, a, b) for a, b in pares]
    # Espera por acción en las combinaciones exclusivas.
    for accion in par.NOMBRES_ACCION:
        filas.append(prueba_pareada(datos, "E2", "E3", grupo=accion))
    return pd.DataFrame(filas)


# ----------------------------------------------------------- teoría vs. sim.

def teoria_vs_simulacion(resumen):
    """Une los valores teóricos con las medias simuladas y sus IC."""
    tabla_teo = teoria.tabla_teoria()
    filas = []
    for _, fila in tabla_teo.iterrows():
        nivel = "global" if fila["grupo"] == "Total" else "caja"
        for metrica_teo, metrica_sim in (("rho", "utilizacion"), ("lq", "lq"),
                                         ("wq", "espera"), ("w", "sistema"),
                                         ("l", "l")):
            sim = resumen[(resumen["escenario"] == fila["escenario"])
                          & (resumen["nivel"] == nivel)
                          & (resumen["grupo"] == fila["grupo"])
                          & (resumen["metrica"] == metrica_sim)]
            if sim.empty:
                continue
            sim = sim.iloc[0]
            filas.append({
                "escenario": fila["escenario"], "grupo": fila["grupo"],
                "modelo": fila["modelo"], "metrica": metrica_teo,
                "teorico": fila[metrica_teo], "simulado": sim["media"],
                "ic_inf": sim["ic_inf"], "ic_sup": sim["ic_sup"],
                "dif_relativa": (sim["media"] - fila[metrica_teo])
                / fila[metrica_teo],
                "teorico_en_ic": bool(sim["ic_inf"] <= fila[metrica_teo]
                                      <= sim["ic_sup"]),
            })
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ Lindley

def _cronometrar(funcion):
    """Ejecuta la función y devuelve (resultado, segundos)."""
    inicio = time.perf_counter()
    resultado = funcion()
    return resultado, time.perf_counter() - inicio


def validar_lindley(num_replicas, semilla=par.SEMILLA,
                    replicas_independientes=10_000):
    """Compara Lindley con eventos discretos y mide los tiempos.

    1. Mismas entradas: las mismas réplicas del experimento se calculan con
       los dos métodos; la diferencia debe ser solo de redondeo.
    2. Semillas independientes: Lindley con otra semilla raíz y muchas más
       réplicas; su media debe ser compatible con el IC de eventos discretos.

    Returns:
        DataFrame con una fila por escenario.
    """
    lote = generar_lote(num_replicas, semilla)
    lote_indep = generar_lote(replicas_independientes, semilla + 1)
    filas = []
    for escenario in par.ESCENARIOS[:4]:
        des, t_des = _cronometrar(
            lambda e=escenario: np.array([simular(c, e).espera.mean()
                                          for c in lote]))
        lin, t_lin = _cronometrar(
            lambda e=escenario: simular_lindley(lote, e).espera_global)
        indep = simular_lindley(lote_indep, escenario).espera_global
        ic_des = intervalo_t(des)
        ic_ind = intervalo_t(indep)
        _, valor_p = stats.ttest_ind(des, indep, equal_var=False)
        filas.append({
            "escenario": escenario.codigo, "replicas": num_replicas,
            "dif_max_mismas_entradas": float(np.max(np.abs(des - lin))),
            "tiempo_des_s": t_des, "tiempo_lindley_s": t_lin,
            "aceleracion": t_des / t_lin,
            "wq_des": ic_des["media"], "wq_des_ic_inf": ic_des["ic_inf"],
            "wq_des_ic_sup": ic_des["ic_sup"],
            "replicas_indep": replicas_independientes,
            "wq_lindley_indep": ic_ind["media"],
            "wq_lindley_indep_me": ic_ind["me"],
            "p_welch": float(valor_p),
        })
    return pd.DataFrame(filas)


def convergencia(datos, escenarios=par.ESCENARIOS_CRITERIO):
    """Media acumulada del Wq global y su IC al aumentar las réplicas."""
    filas = []
    for codigo in escenarios:
        serie = _serie(datos, codigo)
        for num in range(2, len(serie) + 1):
            res = intervalo_t(serie[:num])
            filas.append({"escenario": codigo, "n": num, **res,
                          "er": res["me"] / res["media"]})
    return pd.DataFrame(filas)
