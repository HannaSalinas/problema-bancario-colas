"""Pruebas del generador de clientes, el simulador y la validación Lindley."""

import numpy as np
import pytest

from banco import parametros as par
from banco.clientes import (Clientes, asignar_cajas, flujos_replicas,
                            generar_clientes)
from banco.experimento import correr, generar_lote
from banco.lindley import esperas_lindley, simular_lindley
from banco.simulador import simular


def _clientes(llegada, servicio, accion=None):
    """Construye una secuencia de clientes a mano para casos deterministas."""
    num = len(llegada)
    accion = np.zeros(num, dtype=np.int64) if accion is None else accion
    return Clientes(llegada=np.array(llegada, dtype=float),
                    accion=np.asarray(accion, dtype=np.int64),
                    tipo=np.zeros(num, dtype=np.int64),
                    servicio=np.array(servicio, dtype=float),
                    u_asignacion=np.zeros(num))


UNA_CAJA = par.Escenario("T1", "una caja", (frozenset({0, 1}),))
DOS_CAJAS_FILA_UNICA = par.Escenario("T2", "fila única", (frozenset({0, 1}),) * 2,
                                     fila_unica=True)


# ---------------------------------------------------------------- parámetros

def test_probabilidades_suman_uno():
    """La Tabla 2 y el 70/30 suman 1."""
    assert par.PROB_ACCION.sum() == pytest.approx(1.0)
    assert par.PROB_TIPO.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert par.PROB_CLASE.sum() == pytest.approx(1.0)


def test_escenarios_atienden_ambas_acciones():
    """En todo escenario hay al menos una caja para cada acción."""
    for escenario in par.ESCENARIOS:
        assert escenario.cajas_para(par.RETIRO)
        assert escenario.cajas_para(par.PAGO)


# ------------------------------------------------------------------ clientes

def test_generacion_reproducible():
    """La misma semilla produce la misma secuencia de clientes."""
    uno = generar_clientes(*flujos_replicas(1, 123)[0])
    dos = generar_clientes(*flujos_replicas(1, 123)[0])
    np.testing.assert_array_equal(uno.llegada, dos.llegada)
    np.testing.assert_array_equal(uno.servicio, dos.servicio)


def test_replicas_iniciales_se_conservan_en_corrida_larga():
    """Las primeras réplicas de una corrida larga son las de la corta."""
    corta = generar_lote(3)
    larga = generar_lote(8)
    for c_corta, c_larga in zip(corta, larga):
        np.testing.assert_array_equal(c_corta.llegada, c_larga.llegada)


def test_replicas_distintas():
    """Cada réplica tiene una secuencia diferente."""
    lote = generar_lote(2)
    assert not np.array_equal(lote[0].llegada[:5], lote[1].llegada[:5])


def test_llegadas_ordenadas_y_antes_del_cierre():
    """Las llegadas están ordenadas y ocurren antes de los 480 minutos."""
    for clientes in generar_lote(5):
        assert np.all(np.diff(clientes.llegada) > 0)
        assert clientes.llegada.max() < par.HORIZONTE
        assert np.all(clientes.servicio > 0)


def test_proporciones_de_accion_y_tipo():
    """Con muchas réplicas las proporciones se acercan a las tablas."""
    lote = generar_lote(300)
    accion = np.concatenate([c.accion for c in lote])
    tipo = np.concatenate([c.tipo for c in lote])
    assert np.mean(accion == par.RETIRO) == pytest.approx(0.70, abs=0.01)
    for codigo in (par.RETIRO, par.PAGO):
        frecuencia = np.bincount(tipo[accion == codigo], minlength=4)
        assert frecuencia / frecuencia.sum() == pytest.approx(
            par.PROB_TIPO[codigo], abs=0.015)


def test_tasa_de_llegada_coincide_con_la_teoria():
    """El número medio de llegadas por día es cercano a 480 / E[A]."""
    media = np.mean([len(c) for c in generar_lote(300)])
    esperado = par.HORIZONTE / np.sum(par.PROB_CLASE * par.MEDIA_LLEGADA)
    assert media == pytest.approx(esperado, rel=0.02)


def test_asignacion_solo_a_cajas_que_atienden_la_accion():
    """Cada cliente va a una caja que atiende su acción."""
    clientes = generar_lote(1)[0]
    for escenario in par.ESCENARIOS[:4]:
        caja = asignar_cajas(clientes, escenario)
        for accion in (par.RETIRO, par.PAGO):
            permitidas = set(escenario.cajas_para(accion))
            assert set(caja[clientes.accion == accion]) <= permitidas


def test_asignacion_uniforme():
    """En E1 cada caja recibe cerca de un tercio de los clientes."""
    lote = generar_lote(200)
    cajas = np.concatenate([asignar_cajas(c, par.ESCENARIOS[0]) for c in lote])
    assert np.bincount(cajas) / len(cajas) == pytest.approx([1 / 3] * 3,
                                                            abs=0.01)


# ---------------------------------------------------------------- simulador

def test_caso_determinista_una_caja():
    """Llegadas 0, 1, 2 con servicios 3, 1, 1: esperas 0, 2, 2."""
    resultado = simular(_clientes([0.0, 1.0, 2.0], [3.0, 1.0, 1.0]), UNA_CAJA,
                        horizonte=10.0)
    np.testing.assert_allclose(resultado.espera, [0.0, 2.0, 2.0])
    np.testing.assert_allclose(resultado.fin, [3.0, 4.0, 5.0])
    assert resultado.area_fila[0] == pytest.approx(4.0)
    assert resultado.tiempo_ocupado[0] == pytest.approx(5.0)


def test_caso_determinista_fila_unica():
    """Con dos cajas y fila única el tercer cliente toma la primera libre."""
    resultado = simular(_clientes([0.0, 0.5, 1.0], [4.0, 1.0, 2.0]),
                        DOS_CAJAS_FILA_UNICA, horizonte=10.0)
    np.testing.assert_allclose(resultado.espera, [0.0, 0.0, 0.5])
    np.testing.assert_array_equal(resultado.caja, [0, 1, 1])


def test_cierre_de_puertas_atiende_a_los_que_quedan():
    """Quien llegó antes del cierre se atiende aunque termine después."""
    resultado = simular(_clientes([5.0, 8.0, 9.0], [3.0, 4.0, 4.0]), UNA_CAJA,
                        horizonte=10.0)
    assert resultado.clientes_al_cierre == 2
    assert resultado.ultima_salida == pytest.approx(16.0)
    assert resultado.tiempo_extra == pytest.approx(6.0)
    assert resultado.duracion == pytest.approx(16.0)


def test_sin_tiempo_extra_si_el_banco_queda_vacio():
    """Si todos salen antes del cierre la jornada dura 480 minutos."""
    resultado = simular(_clientes([1.0], [1.0]), UNA_CAJA, horizonte=10.0)
    assert resultado.tiempo_extra == 0.0
    assert resultado.clientes_al_cierre == 0


@pytest.mark.parametrize("escenario", par.ESCENARIOS, ids=lambda e: e.codigo)
def test_invariantes_del_simulador(escenario):
    """Todos se atienden, sin solapes por caja, y se cumple Little."""
    for clientes in generar_lote(5):
        resultado = simular(clientes, escenario)
        assert np.all(resultado.espera >= -1e-12)
        assert np.all(resultado.caja >= 0)
        for k in range(escenario.num_cajas):
            orden = np.argsort(resultado.inicio[resultado.caja == k])
            inicios = resultado.inicio[resultado.caja == k][orden]
            fines = resultado.fin[resultado.caja == k][orden]
            assert np.all(inicios[1:] >= fines[:-1] - 1e-9)
        # La integral de la fila es la suma de las esperas (Lq = λ·Wq).
        assert resultado.area_fila.sum() == pytest.approx(
            resultado.espera.sum())
        assert np.all(resultado.tiempo_ocupado <= resultado.duracion + 1e-9)


def test_fifo_en_cada_caja():
    """En cada caja se atiende en orden de llegada."""
    clientes = generar_lote(1)[0]
    resultado = simular(clientes, par.ESCENARIOS[1])
    for k in range(3):
        inicios = resultado.inicio[resultado.caja == k]
        assert np.all(np.diff(inicios) > 0)


def test_mm1_largo_converge_a_la_teoria():
    """Un M/M/1 puro y largo se acerca a Wq = ρ/(μ-λ)."""
    rng = np.random.default_rng(7)
    num = 200_000
    llegada = np.cumsum(rng.exponential(2.0, num))   # λ = 0,5
    servicio = rng.exponential(1.2, num)              # μ = 1/1,2; ρ = 0,6
    resultado = simular(_clientes(llegada, servicio), UNA_CAJA,
                        horizonte=llegada[-1] + 1)
    teorica = 0.6 / (1 / 1.2 - 0.5)
    assert resultado.espera.mean() == pytest.approx(teorica, rel=0.05)


def test_numeros_comunes_entre_escenarios():
    """Todos los escenarios atienden la misma secuencia de clientes."""
    metricas, _ = correr(3)
    globales = metricas[metricas["nivel"] == "global"]
    conteos = globales.pivot(index="replica", columns="escenario",
                             values="atendidos")
    assert (conteos.nunique(axis=1) == 1).all()


# ------------------------------------------------------------------ Lindley

def test_lindley_caso_determinista():
    """La recursión reproduce el caso calculado a mano."""
    espera = esperas_lindley(np.array([[0.0, 1.0, 2.0]]),
                             np.array([[3.0, 1.0, 1.0]]))
    np.testing.assert_allclose(espera, [[0.0, 2.0, 2.0]])


@pytest.mark.parametrize("escenario", par.ESCENARIOS[:4],
                         ids=lambda e: e.codigo)
def test_lindley_igual_a_eventos_discretos(escenario):
    """Con las mismas entradas Lindley y eventos discretos coinciden."""
    lote = generar_lote(20)
    vectorizado = simular_lindley(lote, escenario)
    for fila, clientes in enumerate(lote):
        resultado = simular(clientes, escenario)
        for k in range(escenario.num_cajas):
            esperas_des = resultado.espera[resultado.caja == k]
            esperas_lin = vectorizado.esperas_por_caja[k][fila]
            esperas_lin = esperas_lin[~np.isnan(esperas_lin)]
            np.testing.assert_allclose(esperas_lin, esperas_des, atol=1e-9)
        assert vectorizado.duracion[fila] == pytest.approx(resultado.duracion)
        np.testing.assert_allclose(
            vectorizado.utilizacion[fila],
            resultado.tiempo_ocupado / resultado.duracion)
        assert vectorizado.espera_global[fila] == pytest.approx(
            resultado.espera.mean())


def test_lindley_rechaza_fila_unica():
    """Lindley no aplica al escenario de fila única."""
    with pytest.raises(ValueError):
        simular_lindley(generar_lote(1), par.escenario_por_codigo("R"))
