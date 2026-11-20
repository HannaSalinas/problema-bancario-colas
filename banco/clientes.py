"""Generación de la secuencia de clientes y asignación a las cajas.

Se usa una sola fila de llegadas, como en la guía del laboratorio: cada cliente
nuevo sortea su acción (retiro 0,7 o pago 0,3), luego su tipo de usuario con la
Tabla 2, luego el tiempo hasta su llegada con la media de llegada de su tipo y
su tiempo de servicio con la media de servicio de su tipo.

Cada réplica usa dos flujos de números aleatorios independientes:
  * flujo de clientes: llegadas, acción, tipo y servicio;
  * flujo de asignación: elección de la caja.
Así la misma secuencia de clientes se conserva en todos los escenarios
(números aleatorios comunes), aunque cambie el número de cajas.
"""

from dataclasses import dataclass

import numpy as np

from banco import parametros as par

# Tamaño de los bloques de números aleatorios. Es fijo para que la secuencia
# generada no dependa de nada más que de la semilla.
TAM_BLOQUE = 512

# Probabilidades acumuladas de la Tabla 2. Se fuerza el último valor a 1 para
# evitar que el redondeo deje un hueco al final.
_ACUMULADA_TIPO = np.cumsum(par.PROB_TIPO, axis=1)
_ACUMULADA_TIPO[:, -1] = 1.0


@dataclass
class Clientes:
    """Clientes que llegan durante la jornada, en orden de llegada."""

    llegada: np.ndarray
    accion: np.ndarray
    tipo: np.ndarray
    servicio: np.ndarray
    u_asignacion: np.ndarray

    def __len__(self):
        return len(self.llegada)


def flujos_replicas(num_replicas, semilla=par.SEMILLA):
    """Crea los generadores de cada réplica a partir de una semilla raíz.

    La semilla raíz es la misma SeedSequence que usa
    np.random.default_rng(semilla). Con SeedSequence.spawn se obtiene un hijo
    por réplica y cada hijo se divide en dos flujos: clientes y asignación.
    Como los hijos se numeran en orden, las primeras réplicas de una corrida
    larga coinciden con las de una corrida corta.

    Devuelve una lista de tuplas (rng_clientes, rng_asignacion).
    """
    raiz = np.random.default_rng(semilla).bit_generator.seed_seq
    flujos = []
    for hijo in raiz.spawn(num_replicas):
        sec_clientes, sec_asignacion = hijo.spawn(2)
        flujos.append((np.random.default_rng(sec_clientes),
                       np.random.default_rng(sec_asignacion)))
    return flujos


def _sortear_tipo(accion, uniformes):
    """Sortea el tipo de usuario por transformada inversa (Tabla 2)."""
    tipo = np.empty(accion.shape, dtype=np.int64)
    for codigo in (par.RETIRO, par.PAGO):
        mascara = accion == codigo
        tipo[mascara] = np.searchsorted(_ACUMULADA_TIPO[codigo],
                                        uniformes[mascara], side="right")
    return np.minimum(tipo, par.PROB_TIPO.shape[1] - 1)


def generar_clientes(rng_clientes, rng_asignacion, horizonte=par.HORIZONTE):
    """Genera los clientes que llegan antes del cierre de puertas.

    Args:
        rng_clientes: generador del flujo de clientes.
        rng_asignacion: generador del flujo de asignación a cajas.
        horizonte: minuto en que se cierran las puertas.

    Returns:
        Objeto Clientes con los arreglos de llegada, acción, tipo, servicio
        y el número uniforme que usará la asignación a caja.
    """
    partes = {"llegada": [], "accion": [], "tipo": [], "servicio": []}
    reloj = 0.0
    while reloj <= horizonte:
        # Se generan los números por bloques para no iterar cliente a cliente.
        accion = (rng_clientes.random(TAM_BLOQUE)
                  >= par.PROB_ACCION[par.RETIRO]).astype(np.int64)
        tipo = _sortear_tipo(accion, rng_clientes.random(TAM_BLOQUE))
        entre_llegadas = (rng_clientes.standard_exponential(TAM_BLOQUE)
                          * par.MEDIA_LLEGADA[accion, tipo])
        servicio = (rng_clientes.standard_exponential(TAM_BLOQUE)
                    * par.MEDIA_SERVICIO[accion, tipo])
        llegada = reloj + np.cumsum(entre_llegadas)
        reloj = llegada[-1]
        partes["llegada"].append(llegada)
        partes["accion"].append(accion)
        partes["tipo"].append(tipo)
        partes["servicio"].append(servicio)

    datos = {clave: np.concatenate(valor) for clave, valor in partes.items()}
    # Solo entran los clientes que llegan antes del cierre.
    dentro = datos["llegada"] < horizonte
    datos = {clave: valor[dentro] for clave, valor in datos.items()}
    u_asignacion = rng_asignacion.random(len(datos["llegada"]))
    return Clientes(u_asignacion=u_asignacion, **datos)


def asignar_cajas(clientes, escenario):
    """Asigna cada cliente a una caja que atienda su acción.

    La caja se escoge con probabilidad uniforme entre las cajas que atienden
    la acción del cliente. Se usa el mismo número uniforme del cliente en
    todos los escenarios. En el escenario de fila única no hay asignación
    previa y se devuelve -1.
    """
    caja = np.full(len(clientes), -1, dtype=np.int64)
    if escenario.fila_unica:
        return caja
    for accion in (par.RETIRO, par.PAGO):
        elegibles = np.array(escenario.cajas_para(accion))
        mascara = clientes.accion == accion
        posicion = np.floor(clientes.u_asignacion[mascara]
                            * len(elegibles)).astype(np.int64)
        caja[mascara] = elegibles[np.minimum(posicion, len(elegibles) - 1)]
    return caja
