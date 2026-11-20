"""Pruebas del módulo de análisis (P1-P5 y comparaciones pareadas)."""

import numpy as np
import pandas as pd
import pytest

from banco import analisis
from banco import teoria
from banco.experimento import correr
from banco.metricas import resumir


@pytest.fixture(scope="module", name="metricas")
def fixture_metricas():
    """Métricas de una corrida corta compartida por las pruebas."""
    datos, _ = correr(15)
    return datos


def test_p1_identifica_menor_y_mayor(metricas):
    """P1 nombra la caja con menor y mayor tiempo de atención."""
    tabla, res = analisis.p1_tiempo_atencion(metricas, "E1")
    assert len(tabla) == 3
    assert res["media_menor"] == pytest.approx(tabla["media"].min())
    assert res["media_mayor"] == pytest.approx(tabla["media"].max())
    assert 0 <= res["p"] <= 1


def test_usuarios_por_tipo_suman_el_total(metricas):
    """Los usuarios por tipo suman el total de atendidos de la réplica."""
    tabla = analisis.usuarios_por_tipo_replica(metricas)
    globales = metricas[(metricas.escenario == "E1")
                        & (metricas.nivel == "global")]
    np.testing.assert_array_equal(tabla["Total"].to_numpy(),
                                  globales.sort_values("replica")
                                  ["atendidos"].to_numpy())


def test_replicas_con_menos_usuarios_reporta_empates():
    """Si dos réplicas empatan en el mínimo se listan ambas."""
    tabla = pd.DataFrame({"Rápido": [5, 3, 3]}, index=[1, 2, 3])
    res = analisis.replicas_con_menos_usuarios(tabla)
    assert res.loc[0, "replicas"] == "2, 3"
    assert res.loc[0, "minimo"] == 3


def test_prueba_pareada_consistente(metricas):
    """La diferencia pareada es la diferencia de medias."""
    res = analisis.prueba_pareada(metricas, "E2", "E3")
    assert res["diferencia"] == pytest.approx(res["media_a"] - res["media_b"])
    assert res["ic_inf"] <= res["diferencia"] <= res["ic_sup"]


def test_teoria_vs_simulacion_tiene_todas_las_metricas(metricas):
    """La comparación incluye ρ, Lq, Wq, W y L para cada escenario."""
    tabla = analisis.teoria_vs_simulacion(resumir(metricas))
    assert set(tabla["metrica"]) == {"rho", "lq", "wq", "w", "l"}
    assert set(tabla["escenario"]) == {"E1", "E2", "E3", "E4", "R"}


def test_coeficientes_de_variacion_mayores_que_uno():
    """Las mezclas de exponenciales son más variables que una exponencial."""
    cv = teoria.coeficientes_variacion()
    assert cv["ca2"] > 1
    assert cv["cs2"] > 1
