"""Programa principal: corre el experimento completo y guarda los resultados.

Uso:
    python -m banco [--salida resultados] [--max-replicas 3000] [--forzar]

Pasos:
  1. Corre n0 = 10 réplicas y calcula cuántas hacen falta para ER < 5 % en
     el Wq global de E1 a E4 (Semana 7).
  2. Corre la cantidad requerida y verifica el ER; si algún escenario no
     cumple, recalcula n con los datos nuevos y vuelve a correr.
  3. Guarda CSV, figuras y reporte.txt.
"""

import argparse
import sys
from pathlib import Path

from banco import analisis, excel, graficas, reporte, teoria
from banco import parametros as par
from banco.experimento import correr
from banco.graficas import num
from banco.metricas import resumir
from banco.replicas import n_final, tabla_replicas


def _argumentos():
    lector = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    lector.add_argument("--salida", default="resultados",
                        help="carpeta de resultados")
    lector.add_argument("--max-replicas", type=int, default=3000,
                        help="detenerse si el n requerido supera este valor")
    lector.add_argument("--forzar", action="store_true",
                        help="correr aunque n supere --max-replicas")
    return lector.parse_args()


def determinar_replicas(max_replicas, forzar):
    """Corrida inicial y ciclo de verificación del número de réplicas.

    Returns:
        Tupla (tabla inicial, lista de (n, tabla de verificación), métricas
        de la corrida final, clientes de las réplicas iniciales) o None si n
        supera el máximo y no se pidió forzar.
    """
    metricas, clientes = correr(par.N_INICIAL, guardar_clientes=True)
    tabla_inicial = tabla_replicas(metricas)
    num_replicas = n_final(tabla_inicial)
    iteraciones = []
    while True:
        if num_replicas > max_replicas and not forzar:
            print(tabla_inicial.to_string())
            print(f"Se requieren {num_replicas} réplicas (> {max_replicas}). "
                  "Use --forzar para continuar.")
            return None
        print(f"Corriendo {num_replicas} réplicas...")
        metricas, _ = correr(num_replicas)
        verificacion = tabla_replicas(metricas)
        iteraciones.append((num_replicas, verificacion))
        if verificacion["cumple"].all():
            return tabla_inicial, iteraciones, metricas, clientes
        num_replicas = max(num_replicas + 1, n_final(verificacion))


def respuestas(ctx):
    """Redacta las respuestas a P1-P5 con las cifras del experimento."""
    resumen = ctx["resumen"]

    def fila_global(codigo, metrica):
        return resumen[(resumen["escenario"] == codigo)
                       & (resumen["nivel"] == "global")
                       & (resumen["metrica"] == metrica)].iloc[0]

    def wq(codigo):
        fila = fila_global(codigo, "espera")
        return (f"{num(fila['media'])} min [{num(fila['ic_inf'])}; "
                f"{num(fila['ic_sup'])}]")

    p1 = ctx["p1"]
    cv = teoria.coeficientes_variacion()
    pruebas = ctx["pruebas"].set_index(["comparacion", "grupo"])
    exclusiva = analisis.mejor_exclusiva(ctx["metricas"])
    e2_e3 = pruebas.loc[("E2 - E3", "Total")]
    e1_ex = pruebas.loc[(f"E1 - {exclusiva}", "Total")]
    e4_e1 = pruebas.loc[("E4 - E1", "Total")]
    r_e1 = pruebas.loc[("R - E1", "Total")]
    erlang = ctx["teoria"][(ctx["teoria"]["escenario"] == "R")
                           & (ctx["teoria"]["metrica"] == "wq")].iloc[0]
    lineas = [
        f"P1. En E1 la caja con menor tiempo promedio de atención es "
        f"{p1['caja_menor']} ({num(p1['media_menor'], 3)} min) y la de mayor "
        f"es {p1['caja_mayor']} ({num(p1['media_mayor'], 3)} min). ANOVA: "
        f"p = {num(p1['p'], 4)}; la diferencia "
        + ("es significativa." if p1["significativa"] else
           "no es significativa: las tres cajas atienden igual y la "
           "diferencia se debe al azar."),
        "P2. Promedio de usuarios por día en el total de cajas: "
        + "; ".join(f"{f['tipo']} {num(f['media'], 1)}"
                    for _, f in ctx["tabla_p2"][ctx["tabla_p2"]["nivel"]
                                                == "tipo"].iterrows()) + ".",
        "P3. Réplica con menos usuarios por tipo: "
        + "; ".join(f"{f['tipo']} réplica {f['replicas']} ({f['minimo']})"
                    for _, f in ctx["menos"].iterrows()) + ".",
        f"P4. Wq global: E1 {wq('E1')}, E4 {wq('E4')}. La cuarta caja reduce "
        f"la espera en {num(-e4_e1['diferencia'])} min (p = "
        f"{e4_e1['p']:.1e}). Utilización en E1: "
        f"{num(fila_global('E1', 'utilizacion')['media'], 3)}.",
        f"P5. E2 {wq('E2')} frente a E3 {wq('E3')}; diferencia E2 - E3 = "
        f"{num(e2_e3['diferencia'])} min [{num(e2_e3['ic_inf'])}; "
        f"{num(e2_e3['ic_sup'])}], p = {e2_e3['p']:.1e}. Mejor exclusiva: "
        f"{exclusiva}. E1 - {exclusiva} = {num(e1_ex['diferencia'])} min "
        f"[{num(e1_ex['ic_inf'])}; {num(e1_ex['ic_sup'])}], p = "
        f"{e1_ex['p']:.1e}.",
        f"Fila única (R): Wq {wq('R')} frente a E1 {wq('E1')}; Erlang C "
        f"M/M/3 = {num(erlang['teorico'], 3)} min. R - E1 = "
        f"{num(r_e1['diferencia'])} min (p = {r_e1['p']:.1e}). Compartir "
        "la fila (pooling) evita que un cliente espere en una caja ocupada "
        "mientras otra está libre; con las mismas 3 cajas y la misma "
        "demanda, la espera baja sin agregar capacidad. R también queda por "
        f"debajo de E4 ({wq('E4')}), que tiene una caja más pero filas "
        "separadas.",
        f"Variabilidad: el servicio de una caja mixta tiene Cs² = "
        f"{num(cv['cs2'], 3)} y el tiempo entre llegadas Ca² = "
        f"{num(cv['ca2'], 3)} (en una exponencial ambos valen 1). Por eso "
        "M/M/1 y Erlang C subestiman la espera, y P-K, que usa el segundo "
        "momento del servicio, se acerca más a lo simulado. Erlang C "
        f"multiplicado por (1 + Cs²)/2 da {num(erlang['teorico'] * (1 + cv['cs2']) / 2, 3)} "
        "min, cercano al Wq simulado de R.",
    ]
    return lineas


def guardar(ctx, carpeta):
    """Guarda CSV, figuras y reporte.txt."""
    carpeta = Path(carpeta)
    csv = carpeta / "csv"
    csv.mkdir(parents=True, exist_ok=True)
    opciones = {"index": False, "encoding": "utf-8", "float_format": "%.6g"}
    ctx["metricas"].to_csv(csv / "metricas_por_replica.csv", **opciones)
    ctx["clientes"].to_csv(csv / "clientes_replicas_iniciales.csv", **opciones)
    ctx["resumen"].to_csv(csv / "resumen_por_escenario.csv", **opciones)
    ctx["teoria"].to_csv(csv / "teoria_vs_simulacion.csv", **opciones)
    ctx["tabla_inicial"].to_csv(csv / "calculo_replicas_inicial.csv",
                                **opciones)
    for numero, (_, tabla) in enumerate(ctx["iteraciones"], start=1):
        tabla.to_csv(csv / f"calculo_replicas_verificacion_{numero}.csv",
                     **opciones)
    ctx["usuarios"].to_csv(csv / "usuarios_por_tipo_replica.csv",
                           encoding="utf-8")
    ctx["usuarios_clase"].to_csv(csv / "usuarios_por_clase_replica.csv",
                                 encoding="utf-8")
    ctx["menos"].to_csv(csv / "replica_con_menos_usuarios.csv", **opciones)
    ctx["tabla_p1"].to_csv(csv / "p1_atencion_por_caja_E1.csv", **opciones)
    ctx["tabla_p2"].to_csv(csv / "p2_promedio_usuarios_tipo.csv", **opciones)
    ctx["pruebas"].to_csv(csv / "pruebas_pareadas.csv", **opciones)
    ctx["lindley"].to_csv(csv / "validacion_lindley.csv", **opciones)
    ctx["convergencia"].to_csv(csv / "convergencia_replicas.csv", **opciones)
    graficas.generar_todas(ctx, carpeta / "figuras")
    excel.exportar(ctx, carpeta / excel.NOMBRE_ARCHIVO)
    texto = reporte.construir(ctx, respuestas(ctx))
    (carpeta / "reporte.txt").write_text(texto, encoding="utf-8")


def main():
    """Punto de entrada."""
    args = _argumentos()
    salida = determinar_replicas(args.max_replicas, args.forzar)
    if salida is None:
        return 1
    tabla_inicial, iteraciones, metricas, clientes = salida
    resumen = resumir(metricas)
    tabla_p1, p1 = analisis.p1_tiempo_atencion(metricas, "E1")
    usuarios = analisis.usuarios_por_tipo_replica(metricas)
    ctx = {
        "n_final": iteraciones[-1][0],
        "tabla_inicial": tabla_inicial,
        "iteraciones": iteraciones,
        "metricas": metricas,
        "clientes": clientes,
        "resumen": resumen,
        "tabla_p1": tabla_p1,
        "p1": p1,
        "p1_otros": {c: analisis.p1_tiempo_atencion(metricas, c)
                     for c in ("E2", "E3", "E4")},
        "tabla_p2": analisis.p2_promedio_usuarios(metricas),
        "usuarios": usuarios,
        "usuarios_clase": analisis.usuarios_por_tipo_replica(metricas, "E1",
                                                             "clase"),
        "menos": analisis.replicas_con_menos_usuarios(usuarios),
        "pruebas": analisis.pruebas_escenarios(metricas),
        "teoria": analisis.teoria_vs_simulacion(resumen),
        "lindley": analisis.validar_lindley(iteraciones[-1][0]),
        "convergencia": analisis.convergencia(metricas),
    }
    guardar(ctx, args.salida)
    print(f"Resultados guardados en {args.salida}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
