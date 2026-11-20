"""Exportación de resultados a un libro de Excel con formato."""

import math

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from banco import parametros as par
from banco import teoria
from banco.metricas import NOMBRES_METRICAS

NOMBRE_ARCHIVO = "Contreras_Hanna_problemaBancario.xlsx"

_ENCABEZADO = Font(bold=True, color="FFFFFF")
_RELLENO = PatternFill("solid", fgColor="1F4E79")
_TITULO = Font(bold=True, size=12, color="1F4E79")
_BORDE = Border(bottom=Side(style="thin", color="BFBFBF"))

NOMBRES_NIVEL = {"caja": "Caja", "clase": "Tipo de usuario (8 clases)",
                 "tipo": "Tipo de usuario", "accion": "Acción",
                 "global": "Global"}


def _tabla(hoja, fila, columnas, registros, formatos=None):
    """Escribe una tabla con encabezado y devuelve la fila siguiente.

    Args:
        hoja: hoja de cálculo.
        fila: fila donde va el encabezado.
        columnas: lista de títulos.
        registros: lista de listas con los valores.
        formatos: diccionario {índice de columna: formato de número}.
    """
    formatos = formatos or {}
    for col, titulo in enumerate(columnas, start=1):
        celda = hoja.cell(row=fila, column=col, value=titulo)
        celda.font = _ENCABEZADO
        celda.fill = _RELLENO
        celda.alignment = Alignment(horizontal="center", vertical="center",
                                    wrap_text=True)
    for desplazamiento, registro in enumerate(registros, start=1):
        for col, valor in enumerate(registro, start=1):
            celda = hoja.cell(row=fila + desplazamiento, column=col,
                              value=_limpiar(valor))
            if col - 1 in formatos:
                celda.number_format = formatos[col - 1]
    return fila + len(registros) + 2


def _limpiar(valor):
    """Convierte tipos de NumPy a tipos nativos y NaN a celda vacía."""
    if hasattr(valor, "item"):
        valor = valor.item()
    if isinstance(valor, float) and math.isnan(valor):
        return None
    return valor


def _titulo(hoja, fila, texto):
    """Escribe un título de sección."""
    hoja.cell(row=fila, column=1, value=texto).font = _TITULO
    return fila + 1


def _ajustar(hoja, minimo=8, maximo=45):
    """Ajusta el ancho de las columnas al contenido."""
    anchos = {}
    for fila in hoja.iter_rows():
        for celda in fila:
            if celda.value is None:
                continue
            largo = len(str(celda.value)) if not isinstance(
                celda.value, float) else 9
            anchos[celda.column] = max(anchos.get(celda.column, 0), largo)
    for col, ancho in anchos.items():
        hoja.column_dimensions[get_column_letter(col)].width = max(
            minimo, min(maximo, ancho + 2))


def _hoja_parametros(libro, ctx):
    hoja = libro.active
    hoja.title = "Parámetros"
    fila = _titulo(hoja, 1, "Tablas 1 y 2 del laboratorio (tiempos en min)")
    registros = []
    for a, accion in enumerate(par.NOMBRES_ACCION):
        for t, tipo in enumerate(par.NOMBRES_TIPO):
            registros.append([accion, tipo, par.PROB_ACCION[a],
                              par.PROB_TIPO[a, t], par.PROB_CLASE[a, t],
                              par.MEDIA_SERVICIO[a, t],
                              par.MEDIA_LLEGADA[a, t]])
    fila = _tabla(hoja, fila, ["Acción", "Tipo de usuario",
                               "Probabilidad de la acción",
                               "Probabilidad del tipo (Tabla 2)",
                               "Probabilidad conjunta",
                               "Media de servicio (min)",
                               "Media de llegada (min)"],
                  registros, {2: "0.00", 3: "0.00", 4: "0.000", 5: "0.0",
                              6: "0.0"})
    media_a, _ = teoria.momentos_llegada()
    caja = teoria.parametros_caja(par.ESCENARIOS[0])[0]
    variacion = teoria.coeficientes_variacion()
    ocho = teoria.carga_ocho_procesos()
    fila = _titulo(hoja, fila, "Parámetros del experimento y valores derivados")
    fila = _tabla(hoja, fila, ["Parámetro", "Valor"], [
        ["Duración de la jornada (min)", par.HORIZONTE],
        ["Semilla raíz", par.SEMILLA],
        ["Nivel de confianza", par.CONFIANZA],
        ["Error relativo objetivo", par.ER_OBJETIVO],
        ["Réplicas iniciales n0", par.N_INICIAL],
        ["Réplicas finales", ctx["n_final"]],
        ["Tiempo medio entre llegadas E[A] (min)", media_a],
        ["Tasa de llegadas λ (clientes/min)", 1 / media_a],
        ["Clientes esperados por día", par.HORIZONTE / media_a],
        ["Tiempo medio de servicio E[S] (min)", caja["es"]],
        ["Ca² del tiempo entre llegadas", variacion["ca2"]],
        ["Cs² del tiempo de servicio", variacion["cs2"]],
        ["Lectura de 8 procesos: λ (clientes/min)", ocho["lambda"]],
        ["Lectura de 8 procesos: carga", ocho["carga"]],
        ["Lectura de 8 procesos: ρ con 3 cajas", ocho["rho"]],
    ], {1: "0.0000"})
    fila = _titulo(hoja, fila, "Escenarios")
    _tabla(hoja, fila, ["Código", "Descripción", "Cajas", "Fila"],
           [[e.codigo, e.descripcion,
             ", ".join(e.etiqueta_caja(k) for k in range(e.num_cajas)),
             "Única compartida" if e.fila_unica else "Una por caja"]
            for e in par.ESCENARIOS])
    _ajustar(hoja)


def _hoja_metricas(libro, titulo, datos, columnas_extra=()):
    hoja = libro.create_sheet(titulo)
    metricas = ["atendidos", "t_atencion", "espera", "sistema", "lq", "l",
                "utilizacion", "espera_max", *columnas_extra]
    columnas = ["Réplica", "Escenario", "Grupo"] + [
        NOMBRES_METRICAS[m] for m in metricas]
    registros = datos[["replica", "escenario", "grupo", *metricas]
                      ].values.tolist()
    formatos = {i: "0.000" for i in range(4, len(columnas))}
    _tabla(hoja, 1, columnas, registros, formatos)
    hoja.freeze_panes = "D2"
    hoja.auto_filter.ref = hoja.dimensions
    _ajustar(hoja, maximo=22)
    hoja.row_dimensions[1].height = 45


def _hoja_p3(libro, ctx):
    hoja = libro.create_sheet("Totales por tipo (P3)")
    usuarios = ctx["usuarios"]
    clases = ctx["usuarios_clase"].drop(columns="Total")
    columnas = ["Réplica", *usuarios.columns, *clases.columns]
    registros = [[replica, *usuarios.loc[replica], *clases.loc[replica]]
                 for replica in usuarios.index]
    _tabla(hoja, 1, columnas, registros)
    hoja.freeze_panes = "B2"
    hoja.auto_filter.ref = f"A1:{get_column_letter(len(columnas))}" \
                           f"{len(registros) + 1}"
    inicio = len(columnas) + 2
    hoja.cell(row=1, column=inicio,
              value="Réplica con menos usuarios por tipo").font = _TITULO
    for col, titulo in enumerate(["Tipo", "Mínimo", "Réplica(s)", "Media",
                                  "Máximo"], start=inicio):
        celda = hoja.cell(row=2, column=col, value=titulo)
        celda.font = _ENCABEZADO
        celda.fill = _RELLENO
    for fila, (_, registro) in enumerate(ctx["menos"].iterrows(), start=3):
        for col, valor in enumerate(registro, start=inicio):
            celda = hoja.cell(row=fila, column=col, value=_limpiar(valor))
            if col == inicio + 3:
                celda.number_format = "0.0"
    _ajustar(hoja, maximo=18)


def _hoja_resumen(libro, ctx):
    hoja = libro.create_sheet("Resumen por escenario")
    resumen = ctx["resumen"]
    registros = [[f["escenario"], NOMBRES_NIVEL[f["nivel"]], f["grupo"],
                  NOMBRES_METRICAS[f["metrica"]], f["n"], f["media"], f["de"],
                  f["me"], f["ic_inf"], f["ic_sup"]]
                 for _, f in resumen.iterrows()]
    _tabla(hoja, 1, ["Escenario", "Nivel", "Grupo", "Métrica", "Réplicas",
                     "Media", "Desviación estándar", "Margen de error",
                     "IC 95 % inferior", "IC 95 % superior"],
           registros, {i: "0.000" for i in range(5, 10)})
    hoja.freeze_panes = "E2"
    hoja.auto_filter.ref = hoja.dimensions
    _ajustar(hoja)


def _hoja_teoria(libro, ctx):
    hoja = libro.create_sheet("Teoría vs. simulación")
    nombres = {"rho": "ρ", "lq": "Lq", "wq": "Wq (min)", "w": "W (min)",
               "l": "L"}
    registros = [[f["escenario"], f["grupo"], f["modelo"],
                  nombres[f["metrica"]], f["teorico"], f["simulado"],
                  f["ic_inf"], f["ic_sup"], f["dif_relativa"],
                  "Sí" if f["teorico_en_ic"] else "No"]
                 for _, f in ctx["teoria"].iterrows()]
    _tabla(hoja, 1, ["Escenario", "Grupo", "Modelo", "Métrica", "Teórico",
                     "Simulado", "IC 95 % inferior", "IC 95 % superior",
                     "Diferencia relativa", "Teórico dentro del IC"],
           registros, {4: "0.000", 5: "0.000", 6: "0.000", 7: "0.000",
                       8: "0.0%"})
    hoja.freeze_panes = "E2"
    hoja.auto_filter.ref = hoja.dimensions
    _ajustar(hoja)


def _hoja_replicas(libro, ctx):
    hoja = libro.create_sheet("Cálculo de réplicas")
    fila = _titulo(hoja, 1, "ER = t(0,025; n-1)·S / (√n · media);  "
                   "n = n0 · (ER0 / 0,05)²  (Semana 7)")
    columnas = ["Escenario", "Réplicas", "Media Wq (min)",
                "Desviación estándar", "t", "Margen de error", "ER",
                "Cumple ER < 5 %", "n por fórmula", "n por t iterativo",
                "n requerido"]

    def registros(tabla):
        return [[f["escenario"], f["n0"], f["media"], f["de"], f["t"],
                 f["me"], f["er"], "Sí" if f["cumple"] else "No",
                 f["n_formula"], f["n_t_iterativo"], f["n_requerido"]]
                for _, f in tabla.iterrows()]

    formatos = {2: "0.000", 3: "0.000", 4: "0.000", 5: "0.000", 6: "0.00%"}
    fila = _titulo(hoja, fila + 1, "Corrida inicial (n0 = 10)")
    fila = _tabla(hoja, fila, columnas, registros(ctx["tabla_inicial"]),
                  formatos)
    for numero, (n_corrida, tabla) in enumerate(ctx["iteraciones"], start=1):
        fila = _titulo(hoja, fila, f"Verificación {numero} con n = {n_corrida}")
        fila = _tabla(hoja, fila, columnas, registros(tabla), formatos)
    _ajustar(hoja, maximo=20)


def _hoja_pruebas(libro, ctx):
    hoja = libro.create_sheet("Pruebas pareadas")
    registros = [[f["comparacion"], f["grupo"], f["media_a"], f["media_b"],
                  f["diferencia"], f["ic_inf"], f["ic_sup"], f["t"], f["p"],
                  "Sí" if f["significativa"] else "No",
                  f["replicas_a_menor"]]
                 for _, f in ctx["pruebas"].iterrows()]
    fila = _tabla(hoja, 1, ["Comparación (A - B)", "Grupo", "Wq A (min)",
                            "Wq B (min)", "Diferencia (min)",
                            "IC 95 % inferior", "IC 95 % superior", "t",
                            "Valor p", "Significativa al 5 %",
                            "Réplicas con A menor"],
                  registros, {2: "0.000", 3: "0.000", 4: "0.000", 5: "0.000",
                              6: "0.000", 7: "0.00", 8: "0.00E+00",
                              10: "0.0%"})
    fila = _titulo(hoja, fila, "P1: ANOVA del tiempo de atención por caja")
    registros = []
    for codigo, (tabla, res) in {"E1": (ctx["tabla_p1"], ctx["p1"]),
                                 **ctx["p1_otros"]}.items():
        for _, f in tabla.iterrows():
            registros.append([codigo, f["caja"], f["media"], f["ic_inf"],
                              f["ic_sup"], res["f"], res["p"]])
    _tabla(hoja, fila, ["Escenario", "Caja", "Tiempo de atención (min)",
                        "IC 95 % inferior", "IC 95 % superior", "F",
                        "Valor p"],
           registros, {2: "0.000", 3: "0.000", 4: "0.000", 5: "0.000",
                       6: "0.0000"})
    _ajustar(hoja)


def _hoja_lindley(libro, ctx):
    hoja = libro.create_sheet("Validación Lindley")
    registros = [[f["escenario"], f["replicas"], f["dif_max_mismas_entradas"],
                  f["tiempo_des_s"], f["tiempo_lindley_s"], f["aceleracion"],
                  f["wq_des"], f["wq_des_ic_inf"], f["wq_des_ic_sup"],
                  f["replicas_indep"], f["wq_lindley_indep"],
                  f["wq_lindley_indep_me"], f["p_welch"]]
                 for _, f in ctx["lindley"].iterrows()]
    _tabla(hoja, 1, ["Escenario", "Réplicas", "Diferencia máxima (min)",
                     "Tiempo eventos discretos (s)", "Tiempo Lindley (s)",
                     "Aceleración", "Wq eventos discretos (min)",
                     "IC 95 % inferior", "IC 95 % superior",
                     "Réplicas independientes", "Wq Lindley independiente",
                     "Margen de error", "Valor p (Welch)"],
           registros, {2: "0.0E+00", 3: "0.000", 4: "0.000", 5: "0.0",
                       6: "0.000", 7: "0.000", 8: "0.000", 10: "0.000",
                       11: "0.000", 12: "0.000"})
    _ajustar(hoja, maximo=20)
    hoja.row_dimensions[1].height = 45


def exportar(ctx, ruta):
    """Crea el libro de Excel con todas las hojas del experimento."""
    libro = Workbook()
    _hoja_parametros(libro, ctx)
    metricas = ctx["metricas"]
    _hoja_metricas(libro, "Datos por réplica",
                   metricas[metricas["nivel"] == "global"],
                   ("al_cierre", "tiempo_extra"))
    _hoja_metricas(libro, "Datos por caja y réplica",
                   metricas[metricas["nivel"] == "caja"])
    _hoja_p3(libro, ctx)
    _hoja_resumen(libro, ctx)
    _hoja_teoria(libro, ctx)
    _hoja_replicas(libro, ctx)
    _hoja_pruebas(libro, ctx)
    _hoja_lindley(libro, ctx)
    for hoja in libro.worksheets:
        for celda in hoja[1]:
            celda.border = _BORDE
    libro.save(ruta)
    return ruta
