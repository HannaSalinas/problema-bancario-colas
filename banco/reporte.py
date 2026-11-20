"""Construcción de reporte.txt con todos los números del experimento."""

import platform
import sys
from importlib import metadata

from banco import parametros as par
from banco import teoria
from banco.graficas import num

LIBRERIAS = ("numpy", "pandas", "scipy", "matplotlib", "openpyxl")


def versiones():
    """Versión de Python y de las librerías usadas."""
    lineas = [f"Python {sys.version.split()[0]} ({platform.system()})"]
    for nombre in LIBRERIAS:
        lineas.append(f"{nombre} {metadata.version(nombre)}")
    return lineas


def _ic(fila, decimales=3):
    """Texto 'media [ic_inf; ic_sup]' con coma decimal."""
    return (f"{num(fila['media'], decimales)} "
            f"[{num(fila['ic_inf'], decimales)}; "
            f"{num(fila['ic_sup'], decimales)}]")


def _titulo(texto):
    return ["", "=" * 78, texto, "=" * 78]


def _seccion_parametros():
    media_a, _ = teoria.momentos_llegada()
    ocho = teoria.carga_ocho_procesos()
    lineas = _titulo("1. PARÁMETROS Y SUPUESTOS")
    lineas += [
        f"Jornada: {num(par.HORIZONTE, 0)} min; cierre de puertas a los "
        f"{num(par.HORIZONTE, 0)} min, se atiende a quien ya entró.",
        f"Semilla raíz: {par.SEMILLA} (SeedSequence.spawn por réplica; "
        "flujos separados para clientes y asignación).",
        f"Tiempo medio entre llegadas E[A] = {num(media_a, 3)} min; "
        f"λ = {num(1 / media_a, 4)} clientes/min "
        f"({num(par.HORIZONTE / media_a, 1)} clientes esperados por día).",
        f"Tiempo medio de servicio E[S] = "
        f"{num(teoria.parametros_caja(par.ESCENARIOS[0])[0]['es'], 3)} min; "
        f"carga total λ·E[S] = "
        f"{num(teoria.tasa_llegada() * teoria.parametros_caja(par.ESCENARIOS[0])[0]['es'] * 3, 3)}"
        " (se divide entre las cajas).",
        "Lectura descartada (8 procesos de llegada independientes): "
        f"λ = {num(ocho['lambda'], 2)} clientes/min, carga = "
        f"{num(ocho['carga'], 2)}, ρ con 3 cajas = {num(ocho['rho'], 2)} > 1 "
        "(inestable).",
    ]
    return lineas


def _seccion_replicas(tabla_inicial, iteraciones):
    lineas = _titulo("2. CÁLCULO DEL NÚMERO DE RÉPLICAS (ER < 5 % EN Wq GLOBAL)")
    lineas.append("Corrida inicial n0 = 10:")
    lineas.append(f"{'Esc.':5}{'media':>9}{'S':>9}{'t':>8}{'ME':>9}"
                  f"{'ER':>9}{'n fórmula':>11}{'n t iter.':>11}")
    for _, fila in tabla_inicial.iterrows():
        lineas.append(f"{fila['escenario']:5}{num(fila['media'], 3):>9}"
                      f"{num(fila['de'], 3):>9}{num(fila['t'], 3):>8}"
                      f"{num(fila['me'], 3):>9}"
                      f"{num(100 * fila['er'], 2) + ' %':>9}"
                      f"{fila['n_formula']:>11}{fila['n_t_iterativo']:>11}")
    for numero, (n_corrida, tabla) in enumerate(iteraciones, start=1):
        lineas.append("")
        lineas.append(f"Verificación {numero} con n = {n_corrida}:")
        for _, fila in tabla.iterrows():
            estado = "cumple" if fila["cumple"] else "no cumple"
            lineas.append(f"  {fila['escenario']}: Wq = {num(fila['media'], 3)}"
                          f" ± {num(fila['me'], 3)} min, ER = "
                          f"{num(100 * fila['er'], 2)} % ({estado})")
    return lineas


def _seccion_resumen(resumen):
    lineas = _titulo("3. RESUMEN GLOBAL POR ESCENARIO (media [IC 95 %])")
    for escenario in par.ESCENARIOS:
        lineas.append(f"{escenario.codigo} - {escenario.descripcion}")
        filas = resumen[(resumen["escenario"] == escenario.codigo)
                        & (resumen["nivel"] == "global")]
        for _, fila in filas.iterrows():
            lineas.append(f"  {fila['metrica']:13}{_ic(fila)}")
    return lineas


def _seccion_niveles(resumen, nivel, titulo):
    lineas = _titulo(titulo)
    for escenario in par.ESCENARIOS:
        filas = resumen[(resumen["escenario"] == escenario.codigo)
                        & (resumen["nivel"] == nivel)]
        lineas.append(f"{escenario.codigo} - {escenario.descripcion}")
        for grupo, datos in filas.groupby("grupo", sort=False):
            valores = {f["metrica"]: f for _, f in datos.iterrows()}
            partes = [f"Wq {_ic(valores['espera'], 2)}",
                      f"aten. {num(valores['t_atencion']['media'], 2)}",
                      f"n {num(valores['atendidos']['media'], 1)}"]
            if "utilizacion" in valores:
                partes.append(f"ρ {num(valores['utilizacion']['media'], 3)}")
            if "lq" in valores:
                partes.append(f"Lq {num(valores['lq']['media'], 3)}")
            partes.append(f"máx {num(valores['espera_max']['media'], 1)}")
            lineas.append(f"  {grupo:18}" + " | ".join(partes))
    return lineas


def _seccion_p1(tabla_p1, p1, tablas_p1_otros):
    lineas = _titulo("P1. TIEMPO PROMEDIO DE ATENCIÓN POR CAJA")
    for _, fila in tabla_p1.iterrows():
        lineas.append(f"  E1 {fila['caja']:16}{_ic(fila)} min")
    lineas.append(f"  Menor: {p1['caja_menor']} ({num(p1['media_menor'], 3)} "
                  f"min); mayor: {p1['caja_mayor']} "
                  f"({num(p1['media_mayor'], 3)} min).")
    lineas.append(f"  ANOVA de un factor: F = {num(p1['f'], 3)}, "
                  f"p = {num(p1['p'], 4)} -> diferencia "
                  f"{'significativa' if p1['significativa'] else 'no significativa'}"
                  " al 5 %.")
    for codigo, (tabla, res) in tablas_p1_otros.items():
        lineas.append(f"  {codigo}: menor {res['caja_menor']} "
                      f"({num(res['media_menor'], 3)}), mayor "
                      f"{res['caja_mayor']} ({num(res['media_mayor'], 3)}); "
                      f"ANOVA p = {num(res['p'], 4)}")
        del tabla
    return lineas


def _seccion_p2_p3(tabla_p2, tabla_usuarios, menos):
    lineas = _titulo("P2. PROMEDIO DE USUARIOS POR TIPO EN EL TOTAL DE CAJAS")
    lineas.append("Con números aleatorios comunes los conteos son iguales en "
                  "todos los escenarios.")
    for _, fila in tabla_p2.iterrows():
        lineas.append(f"  {fila['tipo']:20}{_ic(fila, 2)}  "
                      f"(por caja en E1: {num(fila['por_caja'], 2)})")
    lineas += _titulo("P3. TOTAL DE USUARIOS POR TIPO EN CADA RÉPLICA")
    lineas.append("Réplica  " + "".join(f"{c:>11}" for c in
                                        tabla_usuarios.columns))
    for replica, fila in tabla_usuarios.iterrows():
        lineas.append(f"{replica:7}  " + "".join(f"{v:>11}" for v in fila))
    lineas.append("")
    lineas.append("Réplica con menos usuarios por tipo:")
    for _, fila in menos.iterrows():
        lineas.append(f"  {fila['tipo']:12} mínimo {fila['minimo']:4} en "
                      f"réplica(s) {fila['replicas']} (media "
                      f"{num(fila['media'], 1)}, máximo {fila['maximo']})")
    return lineas


def _seccion_pruebas(pruebas):
    lineas = _titulo("P4 Y P5. PRUEBAS t PAREADAS ENTRE ESCENARIOS (Wq)")
    for _, fila in pruebas.iterrows():
        lineas.append(
            f"  {fila['comparacion']:8} [{fila['grupo']:6}] "
            f"{num(fila['media_a'], 3)} vs {num(fila['media_b'], 3)} min; "
            f"dif. {num(fila['diferencia'], 3)} "
            f"[{num(fila['ic_inf'], 3)}; {num(fila['ic_sup'], 3)}]; "
            f"t = {num(fila['t'], 2)}, p = {fila['p']:.2e}; "
            f"A menor en {num(100 * fila['replicas_a_menor'], 1)} % de las "
            "réplicas")
    return lineas


def _seccion_teoria(tabla_tvs):
    lineas = _titulo("TEORÍA FRENTE A SIMULACIÓN")
    lineas.append("Las fórmulas son de estado estable; la simulación es una "
                  "jornada finita que empieza vacía, con servicio y llegadas "
                  "que son mezclas de exponenciales.")
    for _, fila in tabla_tvs.iterrows():
        lineas.append(
            f"  {fila['escenario']:3}{fila['grupo']:17}{fila['modelo']:6}"
            f"{fila['metrica']:4} teo {num(fila['teorico'], 3):>7}  sim "
            f"{num(fila['simulado'], 3):>7} [{num(fila['ic_inf'], 3)}; "
            f"{num(fila['ic_sup'], 3)}]  dif {num(100 * fila['dif_relativa'], 1)} %")
    return lineas


def _seccion_lindley(validacion):
    lineas = _titulo("VALIDACIÓN CON LA RECURSIÓN DE LINDLEY (VECTORIZADA)")
    for _, fila in validacion.iterrows():
        lineas.append(
            f"  {fila['escenario']}: mismas entradas ({fila['replicas']} rép.)"
            f" dif. máx. {fila['dif_max_mismas_entradas']:.1e} min; tiempo "
            f"DES {num(fila['tiempo_des_s'], 3)} s, Lindley "
            f"{num(fila['tiempo_lindley_s'], 3)} s "
            f"(x{num(fila['aceleracion'], 1)}). Semilla independiente "
            f"({fila['replicas_indep']} rép.): Wq = "
            f"{num(fila['wq_lindley_indep'], 3)} ± "
            f"{num(fila['wq_lindley_indep_me'], 3)} frente a DES "
            f"{num(fila['wq_des'], 3)} [{num(fila['wq_des_ic_inf'], 3)}; "
            f"{num(fila['wq_des_ic_sup'], 3)}]; Welch p = "
            f"{num(fila['p_welch'], 3)}")
    return lineas


def construir(contexto, conclusiones):
    """Arma el texto completo del reporte.

    Args:
        contexto: diccionario con las tablas del análisis.
        conclusiones: lista de líneas con las respuestas redactadas.
    """
    lineas = ["REPORTE DE RESULTADOS - PROBLEMA BANCARIO",
              "Simulación de 3 cajas M/M/1 independientes",
              f"Réplicas finales: {contexto['n_final']}", "",
              "Versiones:"] + [f"  {v}" for v in versiones()]
    lineas += _seccion_parametros()
    lineas += _seccion_replicas(contexto["tabla_inicial"],
                                contexto["iteraciones"])
    lineas += _seccion_resumen(contexto["resumen"])
    lineas += _seccion_niveles(contexto["resumen"], "caja",
                               "MÉTRICAS POR CAJA")
    lineas += _seccion_niveles(contexto["resumen"], "accion",
                               "MÉTRICAS POR ACCIÓN")
    lineas += _seccion_niveles(contexto["resumen"], "clase",
                               "MÉTRICAS POR TIPO DE USUARIO (8 CLASES)")
    lineas += _seccion_p1(contexto["tabla_p1"], contexto["p1"],
                          contexto["p1_otros"])
    lineas += _seccion_p2_p3(contexto["tabla_p2"], contexto["usuarios"],
                             contexto["menos"])
    lineas += _seccion_pruebas(contexto["pruebas"])
    lineas += _seccion_teoria(contexto["teoria"])
    lineas += _seccion_lindley(contexto["lindley"])
    lineas += _titulo("RESPUESTAS")
    lineas += conclusiones
    return "\n".join(lineas) + "\n"
