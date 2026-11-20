"""Parámetros del problema bancario y definición de los escenarios.

Todos los tiempos están en minutos. Los datos vienen de las Tablas 1 y 2 del
laboratorio "Problema Bancario".
"""

from dataclasses import dataclass

import numpy as np

# Semilla raíz del experimento. De ella se derivan las semillas de cada réplica.
SEMILLA = 2026

# Duración de la jornada: 8 horas. A los 480 minutos se cierran las puertas.
HORIZONTE = 480.0

# Códigos de acción.
RETIRO = 0
PAGO = 1
NOMBRES_ACCION = ("Retiro", "Pago")

# Códigos de tipo de usuario.
NOMBRES_TIPO = ("Rápido", "Normal", "Lento", "Muy lento")

# Probabilidad de cada acción (70 % retiros, 30 % pagos).
PROB_ACCION = np.array([0.70, 0.30])

# Tabla 2: probabilidad de cada tipo de usuario dentro de cada acción.
# Fila 0 = retiro, fila 1 = pago; columnas en el orden de NOMBRES_TIPO.
PROB_TIPO = np.array([
    [0.23, 0.40, 0.17, 0.20],
    [0.10, 0.20, 0.30, 0.40],
])

# Tabla 1: media del tiempo de servicio (min), misma disposición que PROB_TIPO.
MEDIA_SERVICIO = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [3.0, 3.0, 5.0, 7.0],
])

# Tabla 1: media del tiempo hasta la llegada (min), misma disposición.
MEDIA_LLEGADA = np.array([
    [1.0, 2.0, 3.0, 3.0],
    [1.0, 2.0, 3.0, 4.0],
])

# Probabilidad conjunta de cada combinación (acción, tipo): 8 clases.
PROB_CLASE = PROB_ACCION[:, None] * PROB_TIPO

# Nivel de confianza de los intervalos y error relativo objetivo (Semana 7).
CONFIANZA = 0.95
ER_OBJETIVO = 0.05
N_INICIAL = 10


@dataclass(frozen=True)
class Escenario:
    """Configuración de cajas de un escenario.

    Atributos:
        codigo: identificador corto (E1, E2, E3, E4 o R).
        descripcion: texto para tablas y figuras.
        cajas: tupla con las acciones que atiende cada caja. Cada elemento es
            un conjunto congelado con RETIRO, PAGO o ambos.
        fila_unica: si es verdadero, todas las cajas comparten una sola fila
            (como el código de la guía, que en realidad es un M/M/c).
    """

    codigo: str
    descripcion: str
    cajas: tuple
    fila_unica: bool = False

    @property
    def num_cajas(self):
        """Número de cajas del escenario."""
        return len(self.cajas)

    def cajas_para(self, accion):
        """Índices de las cajas que atienden la acción indicada."""
        return tuple(i for i, acciones in enumerate(self.cajas)
                     if accion in acciones)

    def etiqueta_caja(self, indice):
        """Nombre legible de una caja, por ejemplo 'Caja 2 (Pago)'."""
        acciones = self.cajas[indice]
        if len(acciones) == 2:
            tipo = "Mixta"
        else:
            tipo = NOMBRES_ACCION[next(iter(acciones))]
        return f"Caja {indice + 1} ({tipo})"


_MIXTA = frozenset({RETIRO, PAGO})
_SOLO_RETIRO = frozenset({RETIRO})
_SOLO_PAGO = frozenset({PAGO})

ESCENARIOS = (
    Escenario("E1", "3 cajas mixtas", (_MIXTA,) * 3),
    Escenario("E2", "1 retiro + 2 pagos",
              (_SOLO_RETIRO, _SOLO_PAGO, _SOLO_PAGO)),
    Escenario("E3", "2 retiros + 1 pago",
              (_SOLO_RETIRO, _SOLO_RETIRO, _SOLO_PAGO)),
    Escenario("E4", "4 cajas mixtas", (_MIXTA,) * 4),
    Escenario("R", "Fila única, 3 cajas (guía)", (_MIXTA,) * 3,
              fila_unica=True),
)

# Escenarios que entran en el criterio de número de réplicas.
ESCENARIOS_CRITERIO = ("E1", "E2", "E3", "E4")


def escenario_por_codigo(codigo):
    """Devuelve el escenario con el código dado."""
    for escenario in ESCENARIOS:
        if escenario.codigo == codigo:
            return escenario
    raise KeyError(f"Escenario desconocido: {codigo}")
