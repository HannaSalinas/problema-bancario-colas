"""Figuras de resultados (PNG a 300 dpi, textos en español)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position
from matplotlib.ticker import FuncFormatter  # pylint: disable=wrong-import-position

from banco import parametros as par  # pylint: disable=wrong-import-position

# Paleta categórica validada (orden fijo) y tintas de texto.
AZUL, NARANJA, AGUAMARINA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA_SECUNDARIA, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLOR_ACCION = {"Retiro": AZUL, "Pago": NARANJA}
COLOR_TIPO_CAJA = {"Retiro": AZUL, "Pago": NARANJA, "Mixta": AGUAMARINA}
COLORES_TIPO = (AZUL, NARANJA, AGUAMARINA, AMARILLO)
DPI = 300

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": REJILLA,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": REJILLA,
    "grid.linewidth": 0.6,
    "xtick.color": TINTA_SECUNDARIA,
    "ytick.color": TINTA_SECUNDARIA,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def num(valor, decimales=2):
    """Número con coma decimal, como se escribe en Colombia."""
    return f"{valor:.{decimales}f}".replace(".", ",")


_FORMATO_EJE = FuncFormatter(lambda x, _: num(x, 1) if x % 1 else f"{x:.0f}")


def _fila(resumen, escenario, nivel, grupo, metrica):
    """Fila del resumen para un escenario, nivel, grupo y métrica."""
    filtro = ((resumen["escenario"] == escenario) & (resumen["nivel"] == nivel)
              & (resumen["grupo"] == grupo) & (resumen["metrica"] == metrica))
    return resumen.loc[filtro].iloc[0]


def _guardar(figura, carpeta, nombre):
    """Guarda la figura a 300 dpi y la cierra."""
    ruta = Path(carpeta) / nombre
    figura.tight_layout()
    figura.savefig(ruta, dpi=DPI)
    plt.close(figura)
    return ruta


def _barras_ic(eje, posiciones, filas, **opciones):
    """Dibuja barras con la media y barras de error con el IC 95 %.

    Las opciones (color, width, label) se pasan a eje.bar.
    """
    medias = np.array([f["media"] for f in filas])
    errores = np.array([f["me"] for f in filas])
    opciones.setdefault("width", 0.6)
    eje.bar(posiciones, medias, edgecolor="white", linewidth=1, **opciones)
    eje.errorbar(posiciones, medias, yerr=errores, fmt="none",
                 ecolor=TINTA, elinewidth=1, capsize=3)
    return medias, errores


def _rotular(eje, posiciones, medias, errores, **opciones):
    """Escribe el valor de cada barra encima de su barra de error."""
    separacion = opciones.get("separacion", 0.08)
    for x, media, error in zip(posiciones, medias, errores):
        eje.text(x, media + error + separacion, num(media), ha="center",
                 va="bottom", color=TINTA, fontsize=opciones.get("tam", 8))


def espera_por_escenario(resumen, carpeta):
    """Wq global de E1-E4 y R con su IC 95 %."""
    codigos = [e.codigo for e in par.ESCENARIOS]
    filas = [_fila(resumen, c, "global", "Total", "espera") for c in codigos]
    figura, eje = plt.subplots(figsize=(6.5, 3.6))
    posiciones = np.arange(len(codigos))
    medias, errores = _barras_ic(eje, posiciones, filas, color=AZUL)
    _rotular(eje, posiciones, medias, errores)
    eje.set_xticks(posiciones, [f"{e.codigo}\n{e.descripcion}"
                                for e in par.ESCENARIOS], fontsize=7.5)
    eje.set_ylabel("Espera promedio en fila Wq (min)")
    eje.set_title("Espera promedio por escenario (IC 95 %)")
    eje.yaxis.set_major_formatter(_FORMATO_EJE)
    return _guardar(figura, carpeta, "fig01_espera_por_escenario.png")


def espera_por_accion(resumen, carpeta, codigos=None, nombre=None):
    """Wq por acción (retiro y pago) en cada escenario con IC 95 %."""
    codigos = codigos or [e.codigo for e in par.ESCENARIOS]
    figura, eje = plt.subplots(figsize=(6.5 if len(codigos) > 2 else 5, 3.6))
    posiciones = np.arange(len(codigos))
    ancho = 0.38
    for desplazamiento, accion in ((-ancho / 2, "Retiro"), (ancho / 2, "Pago")):
        filas = [_fila(resumen, c, "accion", accion, "espera")
                 for c in codigos]
        medias, errores = _barras_ic(eje, posiciones + desplazamiento, filas,
                                     color=COLOR_ACCION[accion], width=ancho,
                                     label=accion)
        _rotular(eje, posiciones + desplazamiento, medias, errores,
                 separacion=0.1, tam=7)
    etiquetas = {e.codigo: f"{e.codigo}\n{e.descripcion}"
                 for e in par.ESCENARIOS}
    eje.set_xticks(posiciones, [etiquetas[c] for c in codigos], fontsize=7.5)
    eje.set_ylabel("Espera promedio en fila Wq (min)")
    eje.set_title("Espera promedio por tipo de acción (IC 95 %)")
    # Leyenda con parches propios (las barras tienen borde blanco) y espacio
    # arriba para que no tape los rótulos.
    eje.legend(handles=[Patch(facecolor=c, label=a)
                        for a, c in COLOR_ACCION.items()],
               loc="upper left", ncols=2)
    eje.set_ylim(0, eje.get_ylim()[1] * 1.15)
    eje.yaxis.set_major_formatter(_FORMATO_EJE)
    return _guardar(figura, carpeta, nombre or "fig02_espera_por_accion.png")


def utilizacion_por_caja(resumen, carpeta):
    """Utilización de cada caja en los escenarios E1-E4 y R."""
    figura, eje = plt.subplots(figsize=(6.5, 3.6))
    x_actual = 0.0
    centros, nombres = [], []
    usados = set()
    for escenario in par.ESCENARIOS:
        inicio = x_actual
        for k in range(escenario.num_cajas):
            etiqueta = escenario.etiqueta_caja(k)
            fila = _fila(resumen, escenario.codigo, "caja", etiqueta,
                         "utilizacion")
            tipo = etiqueta.split("(")[1].rstrip(")")
            leyenda = None if tipo in usados else f"Caja {tipo.lower()}"
            usados.add(tipo)
            eje.bar(x_actual, fila["media"], width=0.8,
                    color=COLOR_TIPO_CAJA[tipo], edgecolor="white",
                    linewidth=1, label=leyenda)
            eje.errorbar(x_actual, fila["media"], yerr=fila["me"], fmt="none",
                         ecolor=TINTA, elinewidth=1, capsize=2)
            eje.text(x_actual, -0.03, str(k + 1), ha="center", va="top",
                     color=TINTA_SECUNDARIA, fontsize=7)
            x_actual += 1
        centros.append((inicio + x_actual - 1) / 2)
        nombres.append(escenario.codigo)
        x_actual += 0.8
    eje.set_xticks(centros, nombres)
    eje.tick_params(axis="x", pad=12, length=0)
    eje.set_ylim(0, 1)
    eje.set_ylabel("Utilización ρ")
    eje.set_title("Utilización por caja (IC 95 %; el número es la caja)")
    eje.yaxis.set_major_formatter(_FORMATO_EJE)
    eje.legend(loc="upper right", ncols=3)
    return _guardar(figura, carpeta, "fig04_utilizacion_por_caja.png")


def convergencia_replicas(tabla_conv, carpeta):
    """Media acumulada del Wq global con banda de IC 95 % (E1-E4)."""
    codigos = list(dict.fromkeys(tabla_conv["escenario"]))
    figura, ejes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True)
    for eje, codigo in zip(ejes.flat, codigos):
        # Se grafica desde las n0 réplicas iniciales; antes el IC es enorme.
        datos = tabla_conv[(tabla_conv["escenario"] == codigo)
                           & (tabla_conv["n"] >= par.N_INICIAL)]
        eje.fill_between(datos["n"], datos["ic_inf"], datos["ic_sup"],
                         color=AZUL, alpha=0.18, linewidth=0)
        eje.plot(datos["n"], datos["media"], color=AZUL, linewidth=1.5)
        final = datos.iloc[-1]
        eje.axhline(final["media"], color=TINTA_SECUNDARIA, linewidth=0.8)
        eje.set_title(f"{codigo}: Wq final = {num(final['media'])} min "
                      f"(ER {num(100 * final['er'], 1)} %)", fontsize=8.5)
        eje.yaxis.set_major_formatter(_FORMATO_EJE)
        eje.set_xscale("log")
        eje.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}"))
    for eje in ejes[1]:
        eje.set_xlabel("Número de réplicas (escala logarítmica)")
    for eje in ejes[:, 0]:
        eje.set_ylabel("Wq acumulada (min)")
    figura.suptitle("Convergencia de la espera promedio desde n0 = 10 "
                    "(banda: IC 95 %)",
                    fontsize=10, fontweight="bold", color=TINTA)
    return _guardar(figura, carpeta, "fig05_convergencia_replicas.png")


def teoria_vs_simulacion(tabla_tvs, carpeta):
    """Wq simulada (IC 95 %) frente a M/M/1, P-K y Erlang C por escenario."""
    tabla = tabla_tvs[(tabla_tvs["grupo"] == "Total")
                      & (tabla_tvs["metrica"] == "wq")]
    codigos = [e.codigo for e in par.ESCENARIOS]
    figura, eje = plt.subplots(figsize=(6.5, 3.6))
    posiciones = np.arange(len(codigos))
    sim = [tabla[tabla["escenario"] == c].iloc[0] for c in codigos]
    # El IC es simétrico: media ± margen de error.
    _barras_ic(eje, posiciones,
               [{"media": f["simulado"], "me": f["ic_sup"] - f["simulado"]}
                for f in sim],
               color=AZUL, width=0.55, label="Simulación")
    marcadores = {"M/M/1": ("o", NARANJA, "M/M/1"),
                  "P-K": ("D", AGUAMARINA, "Pollaczek-Khinchine (M/G/1)"),
                  "M/M/3": ("s", AMARILLO, "Erlang C (M/M/3)")}
    for modelo, (marca, color, etiqueta) in marcadores.items():
        puntos = tabla[tabla["modelo"] == modelo]
        eje.scatter([codigos.index(c) for c in puntos["escenario"]],
                    puntos["teorico"], marker=marca, s=42, color=color,
                    edgecolor="white", linewidth=1, zorder=3, label=etiqueta)
    eje.set_xticks(posiciones, codigos)
    eje.set_ylabel("Espera promedio en fila Wq (min)")
    eje.set_title("Teoría frente a simulación: espera promedio global")
    eje.legend(loc="upper right", fontsize=7.5)
    eje.yaxis.set_major_formatter(_FORMATO_EJE)
    return _guardar(figura, carpeta, "fig06_teoria_vs_simulacion.png")


def _cajas_por_tipo(eje, tabla_usuarios):
    """Diagrama de cajas de los usuarios por tipo en todas las réplicas."""
    cajas = eje.boxplot([tabla_usuarios[t] for t in par.NOMBRES_TIPO],
                        patch_artist=True, widths=0.6,
                        medianprops={"color": TINTA, "linewidth": 1},
                        flierprops={"marker": "o", "markersize": 2,
                                    "markeredgecolor": TINTA_SECUNDARIA})
    for caja, color in zip(cajas["boxes"], COLORES_TIPO):
        caja.set_facecolor(color)
        caja.set_edgecolor(TINTA_SECUNDARIA)
    eje.set_xticks(range(1, 5), ["Ráp.", "Nor.", "Len.", "M. len."])
    eje.set_title(f"Todas ({len(tabla_usuarios)} réplicas)", fontsize=9)


def usuarios_por_replica(tabla_usuarios, carpeta, iniciales=par.N_INICIAL):
    """Usuarios de cada tipo por réplica.

    Panel izquierdo: las réplicas iniciales, una barra por tipo. Panel
    derecho: distribución de los conteos en todas las réplicas. La tabla
    completa está en el CSV y en el Excel.
    """
    figura, (izq, der) = plt.subplots(
        1, 2, figsize=(6.5, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    primeras = tabla_usuarios.head(iniciales)
    posiciones = np.arange(len(primeras))
    ancho = 0.2
    for i, (tipo, color) in enumerate(zip(par.NOMBRES_TIPO, COLORES_TIPO)):
        izq.bar(posiciones + (i - 1.5) * ancho, primeras[tipo], width=ancho,
                color=color, edgecolor="white", linewidth=0.6, label=tipo)
    izq.set_xticks(posiciones, primeras.index)
    izq.set_xlabel("Réplica")
    izq.set_ylabel("Usuarios atendidos en el día")
    izq.set_title(f"Primeras {len(primeras)} réplicas", fontsize=9)
    _cajas_por_tipo(der, tabla_usuarios)
    figura.legend(*izq.get_legend_handles_labels(), loc="lower center",
                  ncols=4, bbox_to_anchor=(0.5, 0.0))
    figura.suptitle("Usuarios por tipo en cada réplica", fontsize=10,
                    fontweight="bold", color=TINTA)
    figura.tight_layout(rect=(0, 0.07, 1, 1))
    ruta = Path(carpeta) / "fig07_usuarios_por_tipo_replica.png"
    figura.savefig(ruta, dpi=DPI)
    plt.close(figura)
    return ruta


def atencion_por_caja(tabla_p1, carpeta):
    """Tiempo promedio de atención por caja en E1 con IC 95 % (P1)."""
    figura, eje = plt.subplots(figsize=(5, 3.4))
    posiciones = np.arange(len(tabla_p1))
    filas = [fila for _, fila in tabla_p1.iterrows()]
    medias, errores = _barras_ic(eje, posiciones, filas, color=AGUAMARINA)
    _rotular(eje, posiciones, medias, errores, separacion=0.03)
    eje.set_xticks(posiciones, tabla_p1["caja"])
    eje.set_ylabel("Tiempo promedio de atención (min)")
    eje.set_title("E1: tiempo promedio de atención por caja (IC 95 %)")
    eje.yaxis.set_major_formatter(_FORMATO_EJE)
    return _guardar(figura, carpeta, "fig08_atencion_por_caja_E1.png")


def generar_todas(ctx, carpeta):
    """Genera todas las figuras y devuelve la lista de rutas.

    Args:
        ctx: diccionario del análisis con las claves resumen, convergencia,
            teoria, usuarios y tabla_p1.
        carpeta: carpeta de salida.
    """
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    resumen = ctx["resumen"]
    return [
        espera_por_escenario(resumen, carpeta),
        espera_por_accion(resumen, carpeta),
        espera_por_accion(resumen, carpeta, ["E2", "E3"],
                          "fig03_espera_por_accion_E2_E3.png"),
        utilizacion_por_caja(resumen, carpeta),
        convergencia_replicas(ctx["convergencia"], carpeta),
        teoria_vs_simulacion(ctx["teoria"], carpeta),
        usuarios_por_replica(ctx["usuarios"], carpeta),
        atencion_por_caja(ctx["tabla_p1"], carpeta),
    ]
