"""Simulación de eventos discretos de un día de operación del banco.

Hay dos tipos de evento: llegada de un cliente y fin de servicio en una caja.
La lista de eventos futuros es un montículo binario (heapq), así que sacar el
evento más próximo cuesta O(log n) en lugar de ordenar la lista en cada paso.
Cada fila es una deque, que saca al primer cliente en O(1) en lugar del
pop(0) de una lista, que cuesta O(n).

Reglas del modelo:
  * En los escenarios E1 a E4 cada caja tiene su propia fila (M/M/1
    independientes). El cliente llega con su caja ya asignada.
  * En el escenario R todas las cajas comparten una sola fila, como en el
    código de la guía. El cliente pasa a la primera caja libre (la de menor
    índice), igual que en la guía.
  * A los 480 minutos se cierran las puertas: no entra nadie más, pero se
    atiende a todos los que ya están dentro.
"""

import heapq
import itertools
from collections import deque
from dataclasses import dataclass

import numpy as np

from banco import parametros as par
from banco.clientes import Clientes, asignar_cajas

# Prioridad de los eventos cuando coinciden en el tiempo: primero se libera
# la caja y luego entra el cliente.
SALIDA = 0
LLEGADA = 1


@dataclass
class ResultadoDia:
    """Resultado de simular un día en un escenario."""

    escenario: par.Escenario
    clientes: Clientes
    caja: np.ndarray
    inicio: np.ndarray
    fin: np.ndarray
    tiempo_ocupado: np.ndarray
    area_fila: np.ndarray
    horizonte: float

    @property
    def espera(self):
        """Tiempo en fila de cada cliente (Wq individual)."""
        return self.inicio - self.clientes.llegada

    @property
    def sistema(self):
        """Tiempo total en el banco de cada cliente (W individual)."""
        return self.fin - self.clientes.llegada

    @property
    def ultima_salida(self):
        """Minuto en que sale el último cliente."""
        return float(self.fin.max()) if len(self.fin) else 0.0

    @property
    def duracion(self):
        """Duración real de la jornada: 480 minutos más el tiempo extra."""
        return max(self.horizonte, self.ultima_salida)

    @property
    def tiempo_extra(self):
        """Minutos que se trabajaron después del cierre de puertas."""
        return self.duracion - self.horizonte

    @property
    def clientes_al_cierre(self):
        """Clientes que seguían en el banco cuando se cerraron las puertas."""
        return int(np.sum(self.fin > self.horizonte))


class _Simulacion:  # pylint: disable=too-few-public-methods
    """Estado de un día simulado: reloj, eventos, filas y cajas."""

    def __init__(self, clientes, escenario):
        self.escenario = escenario
        self.llegada = clientes.llegada.tolist()
        self.servicio = clientes.servicio.tolist()
        self.caja = asignar_cajas(clientes, escenario).tolist()
        # Instantes de inicio y fin de servicio de cada cliente.
        self.tiempos = {"inicio": [0.0] * len(clientes),
                        "fin": [0.0] * len(clientes)}
        num_filas = 1 if escenario.fila_unica else escenario.num_cajas
        self.filas = [deque() for _ in range(num_filas)]
        self.ocupada = [False] * escenario.num_cajas
        # Acumuladores: integral de la longitud de cada fila, último cambio
        # de cada fila y tiempo ocupado de cada caja.
        self.acumulado = {"area": [0.0] * num_filas,
                          "ultimo": [0.0] * num_filas,
                          "ocupado": [0.0] * escenario.num_cajas}
        self.eventos = []
        self.secuencia = itertools.count()      # desempata eventos simultáneos

    def _programar(self, tiempo, tipo_evento, dato):
        """Agrega un evento a la lista de eventos futuros."""
        heapq.heappush(self.eventos,
                       (tiempo, tipo_evento, next(self.secuencia), dato))

    def _acumular_area(self, fila, reloj):
        """Suma el área bajo la longitud de la fila hasta el reloj actual."""
        acumulado = self.acumulado
        acumulado["area"][fila] += (len(self.filas[fila])
                                    * (reloj - acumulado["ultimo"][fila]))
        acumulado["ultimo"][fila] = reloj

    def _iniciar_servicio(self, cliente, indice_caja, reloj):
        """Pone al cliente en la caja y programa su fin de servicio."""
        fin = reloj + self.servicio[cliente]
        self.tiempos["inicio"][cliente] = reloj
        self.tiempos["fin"][cliente] = fin
        self.caja[cliente] = indice_caja
        self.ocupada[indice_caja] = True
        self.acumulado["ocupado"][indice_caja] += self.servicio[cliente]
        self._programar(fin, SALIDA, indice_caja)

    def _llegada(self, cliente, reloj):
        """Procesa la llegada de un cliente."""
        # La llegada siguiente se programa al procesar la actual.
        if cliente + 1 < len(self.llegada):
            self._programar(self.llegada[cliente + 1], LLEGADA, cliente + 1)
        if self.escenario.fila_unica:
            libre = next((k for k, ocupada in enumerate(self.ocupada)
                          if not ocupada), None)
            fila = 0
        else:
            fila = self.caja[cliente]
            libre = None if self.ocupada[fila] else fila
        if libre is not None:
            self._iniciar_servicio(cliente, libre, reloj)
        else:
            self._acumular_area(fila, reloj)
            self.filas[fila].append(cliente)

    def _salida(self, indice_caja, reloj):
        """Procesa el fin de servicio en una caja."""
        self.ocupada[indice_caja] = False
        fila = 0 if self.escenario.fila_unica else indice_caja
        if self.filas[fila]:
            self._acumular_area(fila, reloj)
            self._iniciar_servicio(self.filas[fila].popleft(), indice_caja,
                                   reloj)

    def correr(self):
        """Procesa eventos hasta que no quede nadie en el banco."""
        if self.llegada:
            self._programar(self.llegada[0], LLEGADA, 0)
        while self.eventos:
            reloj, tipo_evento, _, dato = heapq.heappop(self.eventos)
            if tipo_evento == LLEGADA:
                self._llegada(dato, reloj)
            else:
                self._salida(dato, reloj)


def simular(clientes, escenario, horizonte=par.HORIZONTE):
    """Simula un día completo del escenario con la secuencia de clientes dada.

    Args:
        clientes: objeto Clientes generado con generar_clientes.
        escenario: configuración de cajas.
        horizonte: minuto de cierre de puertas. Las llegadas ya vienen
            filtradas; aquí solo se usa para medir el tiempo extra.

    Returns:
        ResultadoDia con los tiempos de cada cliente y los acumulados por caja.
    """
    estado = _Simulacion(clientes, escenario)
    estado.correr()
    return ResultadoDia(
        escenario=escenario,
        clientes=clientes,
        caja=np.array(estado.caja, dtype=np.int64),
        inicio=np.array(estado.tiempos["inicio"]),
        fin=np.array(estado.tiempos["fin"]),
        tiempo_ocupado=np.array(estado.acumulado["ocupado"]),
        area_fila=np.array(estado.acumulado["area"]),
        horizonte=horizonte,
    )
