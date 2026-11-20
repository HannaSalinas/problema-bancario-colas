"""Pruebas de las fórmulas teóricas, los intervalos y el cálculo de réplicas."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from banco import parametros as par
from banco import teoria
from banco.experimento import correr
from banco.metricas import intervalo_t, resumir
from banco.replicas import (n_final, n_t_iterativo, replicas_por_formula,
                            tabla_replicas)


# -------------------------------------------------------------------- teoría

def test_mm1_ejemplo_semana_6():
    """Ejemplo del cajero de la Semana 6: λ = 8/h, μ = 12/h."""
    res = teoria.mm1(8.0, 1 / 12)
    assert res["rho"] == pytest.approx(2 / 3)
    assert res["l"] == pytest.approx(2.0)
    assert res["lq"] == pytest.approx(4 / 3)
    assert res["w"] == pytest.approx(0.25)
    assert res["wq"] == pytest.approx(1 / 6)


def test_mm1_inestable():
    """Con ρ ≥ 1 las métricas son infinitas."""
    assert teoria.mm1(2.0, 1.0)["wq"] == np.inf


def test_pk_con_servicio_exponencial_es_mm1():
    """Si E[S²] = 2E[S]², Pollaczek-Khinchine coincide con M/M/1."""
    pk = teoria.mg1(0.5, 1.2, 2 * 1.2 ** 2)
    mm1 = teoria.mm1(0.5, 1.2)
    for clave in ("rho", "lq", "wq", "w", "l"):
        assert pk[clave] == pytest.approx(mm1[clave])


def test_mmc_con_un_servidor_es_mm1():
    """Erlang C con c = 1 coincide con M/M/1."""
    mmc = teoria.mmc(0.5, 1.2, 1)
    mm1 = teoria.mm1(0.5, 1.2)
    assert mmc["wq"] == pytest.approx(mm1["wq"])
    assert mmc["l"] == pytest.approx(mm1["l"])


def test_mmc_valor_conocido():
    """M/M/2 con λ = 1 y μ = 1: Lq = 1/3."""
    assert teoria.mmc(1.0, 1.0, 2)["lq"] == pytest.approx(1 / 3)


def test_momentos_del_problema():
    """E[A] = 2,398 min y E[S] = 3,198 min con las Tablas 1 y 2."""
    media_llegada, _ = teoria.momentos_llegada()
    assert media_llegada == pytest.approx(2.398)
    caja = teoria.parametros_caja(par.ESCENARIOS[0])[0]
    assert caja["es"] == pytest.approx(3.198)


def test_carga_de_ocho_procesos():
    """La lectura de 8 procesos da una carga de 12,25 y ρ > 4 con 3 cajas."""
    res = teoria.carga_ocho_procesos(3)
    assert res["lambda"] == pytest.approx(4.25)
    assert res["carga"] == pytest.approx(12.25)
    assert res["rho"] > 1


@pytest.mark.parametrize("escenario", par.ESCENARIOS[:4],
                         ids=lambda e: e.codigo)
def test_tasas_por_caja_suman_la_total(escenario):
    """La suma de las tasas de las cajas es la tasa total de llegadas."""
    cajas = teoria.parametros_caja(escenario)
    assert sum(c["lambda"] for c in cajas) == pytest.approx(
        teoria.tasa_llegada())


def test_tabla_teoria_estable():
    """En todos los escenarios ρ < 1."""
    tabla = teoria.tabla_teoria()
    assert (tabla["rho"] < 1).all()
    assert set(tabla["escenario"]) == {e.codigo for e in par.ESCENARIOS}


# --------------------------------------------------------------- intervalos

def test_intervalo_coincide_con_scipy():
    """El IC coincide con scipy.stats.t.interval (datos de la Semana 7)."""
    datos = [8.5, 9.2, 7.8, 8.9, 9.5, 8.1, 8.7, 9.3, 8.4, 9.0, 8.6, 9.1]
    res = intervalo_t(datos)
    inferior, superior = stats.t.interval(0.95, len(datos) - 1,
                                          loc=np.mean(datos),
                                          scale=stats.sem(datos))
    assert res["media"] == pytest.approx(8.758, abs=1e-3)
    assert res["ic_inf"] == pytest.approx(inferior)
    assert res["ic_sup"] == pytest.approx(superior)


def test_margen_de_error_ejemplo_semana_7():
    """Con X̄ = 8,758, S = 0,540 y n = 12 la Semana 7 obtiene ME = 0,343."""
    t_critico = stats.t.ppf(0.975, 11)
    assert t_critico == pytest.approx(2.201, abs=1e-3)
    assert t_critico * 0.540 / np.sqrt(12) == pytest.approx(0.343, abs=1e-3)


def test_intervalo_ignora_nan_y_casos_pequenos():
    """Los NaN se descartan y con un dato no hay intervalo."""
    assert intervalo_t([1.0, np.nan, 3.0])["media"] == pytest.approx(2.0)
    assert np.isnan(intervalo_t([1.0])["me"])
    assert intervalo_t([])["n"] == 0


def test_resumir_estructura():
    """El resumen tiene una fila por escenario, nivel, grupo y métrica."""
    metricas, _ = correr(3)
    resumen = resumir(metricas)
    assert {"media", "de", "ic_inf", "ic_sup"} <= set(resumen.columns)
    fila = resumen[(resumen.escenario == "E1") & (resumen.nivel == "global")
                   & (resumen.metrica == "espera")]
    assert len(fila) == 1
    assert fila["n"].iloc[0] == 3


# ------------------------------------------------------------------ réplicas

def _datos_globales(valores, escenario="E1"):
    """DataFrame mínimo con el formato de las métricas por réplica."""
    return pd.DataFrame({"replica": range(1, len(valores) + 1),
                         "escenario": escenario, "nivel": "global",
                         "grupo": "Total", "espera": valores})


def test_replicas_ejemplo_semana_7():
    """Call center de la Semana 7: X̄ = 5,65, S = 0,487, n0 = 10 → n = 16."""
    error, num = replicas_por_formula(10, 5.65, 0.487)
    assert error == pytest.approx(0.0616, abs=5e-4)
    assert num == 16


def test_tabla_replicas_con_datos():
    """La tabla calcula el ER a partir de los datos por réplica."""
    valores = [5.2, 6.1, 4.8, 5.7, 6.3, 5.1, 5.9, 6.2, 5.4, 5.8]
    tabla = tabla_replicas(_datos_globales(valores), escenarios=("E1",))
    esperado = (stats.t.ppf(0.975, 9) * np.std(valores, ddof=1)
                / (np.sqrt(10) * np.mean(valores)))
    assert tabla["er"].iloc[0] == pytest.approx(esperado)
    assert tabla["n_formula"].iloc[0] == int(np.ceil(10 * (esperado / 0.05)
                                                     ** 2))
    assert not tabla["cumple"].iloc[0]


def test_n_t_iterativo_cumple_el_error():
    """El n del método iterativo cumple el ER y n - 1 no lo cumple."""
    num = n_t_iterativo(5.65, 0.487, er_objetivo=0.05)

    def error(n):
        return stats.t.ppf(0.975, n - 1) * 0.487 / (np.sqrt(n) * 5.65)
    assert error(num) <= 0.05 < error(num - 1)


def test_n_final_es_el_maximo_con_minimo_diez():
    """El n final es el máximo de los escenarios y nunca menor que 10."""
    datos = pd.concat([
        _datos_globales([5.0, 5.01, 4.99, 5.0, 5.02, 4.98, 5.0, 5.01, 4.99,
                         5.0], "E1"),
        _datos_globales([5.2, 6.1, 4.8, 5.7, 6.3, 5.1, 5.9, 6.2, 5.4, 5.8],
                        "E2"),
    ])
    tabla = tabla_replicas(datos, escenarios=("E1", "E2"))
    assert tabla.loc[0, "n_requerido"] == 10
    assert n_final(tabla) == tabla.loc[1, "n_requerido"] > 10
