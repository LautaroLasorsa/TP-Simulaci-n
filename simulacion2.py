import numpy as np
import networkx as nx
import random
from typing import List, Tuple

class Simulacion:
    def __init__(self, n: int, m: int, til_base: int, tic_base: int, 
                 memoria: float, max_tolerancia_riesgo: float, umbral: float, fun_contagio , initial_exposeds : int = 1):
        """
        Inicializa una instancia de la clase Simulacion.

        Parámetros:
        n (int): Número de personas.
        m (int): Número de entidades.
        til_base (int): Tiempo base que dura el período de enfermedad en latencia. Luego de este tiempo, en cada día el individuo tiene una probabilidad de 0.5 de pasar al estado infectado.
        tic_base (int): Tiempo base que dura el periodo de enfermedad contagiosa. Luego de este tiempo, en cada día el individuo tiene una probabilidad de 0.5 de recuperarse.
        memoria (float): Coeficiente que se utiliza para actualizar el riesgo percibido para realizar una determinada actividad.
        max_tolerancia_riesgo (float): Máximo valor que puede tomar la tolerancia al riesgo de un individuo.
        umbral (float): Umbral de corte de satisfacción para inicializar el grafo.
        beta (float): Riesgo de contagio. Una persona expuesta a un lugar con A expuestos y B asistentes sigue sana con probabilidad exp(-A/(beta*B)).
        min_asistencias_para_cerrar (int): Número mínimo de asistencias diarias para mantener una entidad abierta.
        max_infecciones_para_evitar (int): Número máximo de infecciones que puede tener un individuo en una entidad antes de evitarla.
        politicas_publicas (bool): Indica si se implementan políticas públicas en la simulación.
        initial_exposeds (int): Cantidad de invididuos expuestos al inicio de la simulación.
        """
        self.n = n
        self.m = m
        self.til_base = til_base
        self.tic_base = tic_base
        self.memoria = memoria
        self.max_tolerancia_riesgo = max_tolerancia_riesgo
        self.umbral = umbral
        self.fun_contagio = fun_contagio
        self.initial_exposeds = initial_exposeds
        
        self.B = self.inicializar_grafo()
        self.estados, self.duraciones = self.inicializar_estados_duracion()
        self.tolerancias = self.inicializar_tolerancias()
        self.infectados_por_entidad = {i: 0 for i in range(n, n + m)}
        self.expuestos_por_entidad = {i: 0 for i in range(n, n + m)}
        self.asistencias_por_entidad = {i: 0 for i in range(n, n + m)}
        self.individuo_asistio_a_entidad = np.zeros((n, m))
        self.riesgo_por_entidad = {i: 0 for i in range(n, n + m)}

    def asistencias_posibles(self):
        return self.B.edges()

    def inicializar_grafo(self) -> nx.Graph:
        """
        Inicializa el grafo bipartito pesado.

        Retorna:
        nx.Graph: Grafo bipartito pesado con pesos generados según una distribución lognormal.
        """
        B = nx.Graph()
        personas = range(self.n)
        entidades = range(self.n, self.n + self.m)
        B.add_nodes_from(personas, bipartite=0)
        B.add_nodes_from(entidades, bipartite=1)
        
        matriz = np.random.lognormal(0, 1, (self.n, self.m))
        for i in personas:
            copia = matriz[i].copy()
            copia.sort()
            umbral = min(self.umbral, copia[-3])
            matriz[i] -= umbral
            suma = sum([matriz[i][j] for j in range(self.m) if matriz[i][j] > 0])
            for j in range(self.m):
                matriz[i][j] /= suma
            for j in entidades:
                if matriz[i][j - self.n] > 0:
                    B.add_edge(i, j, satisfaccion=matriz[i][j - self.n])
        return B

    def inicializar_estados_duracion(self) -> Tuple[List[str], List[int]]:
        """
        Inicializa los estados y las duraciones de los individuos.

        Retorna:
        Tuple[List[str], List[int]]: Lista de estados y lista de duraciones de cada individuo.
        """
        estados = ['S'] * self.n
        duraciones = [0] * self.n
        for i in range(self.initial_exposeds):
            estados[i] = 'E'
            duraciones[i] = self.til_base + np.random.geometric(0.5)
        # initial_exposed = random.randint(0, self.n-1)
        # estados[initial_exposed] = 'E'
        # duraciones[initial_exposed] = self.til_base + np.random.geometric(0.5)
        return estados, duraciones

    def inicializar_tolerancias(self) -> List[float]:
        """
        Inicializa las tolerancias al riesgo de los individuos.

        Retorna:
        List[float]: Lista de tolerancias al riesgo de cada individuo.
        """
        return [random.uniform(0, self.max_tolerancia_riesgo) for _ in range(self.n)]

    def calcular_riesgo(self, entidad: int) -> float:
        """
        Calcula el riesgo de asistir a una entidad específica.
        AGREGAR: asistencia = asistencia * memoria + (1-memoria)*np.exp(- (exp*etoi) ** alfa * beta)

        Parámetros:
        entidad (int): ID de la entidad.

        Retorna:
        float: Riesgo calculado para la entidad.
        """
        if self.asistencias_por_entidad[entidad] == 0:
            return 0.0
        infectados_reportados = self.infectados_por_entidad[entidad]
        asistentes = self.asistencias_por_entidad[entidad]
        return self.memoria * self.riesgo_por_entidad[entidad] + (1-self.memoria)*(infectados_reportados / asistentes)

    def actualizar_riesgos(self) -> None:
        """
        Actualiza los riesgos de las entidades.
        """
        for entidad in range(self.n, self.n + self.m):
            self.riesgo_por_entidad[entidad] = self.calcular_riesgo(entidad)

    def decidir_asistencia(self, tolerancia: float, riesgo: float, satisfaccion: float) -> bool:
        """
        Determina si un individuo decide asistir a una entidad.

        Parámetros:
        tolerancia (float): Tolerancia al riesgo del individuo.
        riesgo (float): Riesgo percibido de asistir a la entidad.
        satisfaccion (float): Satisfacción de asistir a la entidad.

        Retorna:
        bool: True si el individuo decide asistir, False en caso contrario.
        """
        return riesgo * satisfaccion < tolerancia

    def actualizar_estados_duracion(self) -> None:
        """
        Actualiza los estados y las duraciones de los individuos.
        """
        nuevos_estados = self.estados.copy()
        nuevas_duraciones = self.duraciones.copy()
        for i, estado in enumerate(self.estados):
            if estado == 'S':
                for entidad in self.B.neighbors(i):
                    contagios = self.expuestos_por_entidad[entidad]
                    probabilidad_contagio = 0.0 if self.asistencias_por_entidad[entidad] == 0 else self.fun_contagio(contagios, self.asistencias_por_entidad[entidad])#1 - np.exp(-contagios / (self.beta * self.asistencias_por_entidad[entidad]))
                    if self.individuo_asistio_a_entidad[i][entidad - self.n]:
                        if random.random() < probabilidad_contagio:
                            nuevos_estados[i] = 'E'
                            nuevas_duraciones[i] = self.til_base + np.random.geometric(0.5)
            elif estado == 'E':
                if self.duraciones[i] > 0:
                    nuevas_duraciones[i] -= 1
                else:
                    nuevos_estados[i] = 'I'
                    nuevas_duraciones[i] = self.tic_base + np.random.geometric(0.5)
            elif estado == 'I':
                if self.duraciones[i] > 0:
                    nuevas_duraciones[i] -= 1
                else:
                    nuevos_estados[i] = 'S'
                    nuevas_duraciones[i] = 0
        self.estados = nuevos_estados
        self.duraciones = nuevas_duraciones

    def registrar_asistencias(self) -> None:
        """
        Registra las asistencias y actualiza los contadores de infectados reportados en el turno anterior y asistentes por entidad.
        """
        self.infectados_por_entidad = {i: 0 for i in range(self.n, self.n + self.m)}
        self.asistencias_por_entidad = {i: 0 for i in range(self.n, self.n + self.m)}
        self.expuestos_por_entidad = {i: 0 for i in range(self.n, self.n + self.m)}
        self.individuo_asistio_a_entidad = np.zeros((self.n, self.m))

        for i in range(self.n):
            for entidad in self.B.neighbors(i):
                
                acude = (self.estados[i] in "ES") and self.decidir_asistencia(
                    self.tolerancias[i], 
                    self.riesgo_por_entidad[entidad], 
                    self.B[i][entidad]['satisfaccion']
                )
                if acude:
                    self.asistencias_por_entidad[entidad] += 1
                    self.individuo_asistio_a_entidad[i][entidad - self.n] = 1
                    if self.estados[i] == 'E' and self.duraciones[i]==0:
                        self.infectados_por_entidad[entidad] += 1
                    elif self.estados[i] == 'E':
                        self.expuestos_por_entidad[entidad] += 1

    def ejecutar_simulacion(self, dias: int) -> Tuple[List[int], List[float], List[int]]:
        """
        Ejecuta la simulación durante un número de días especificado.

        Parámetros:
        dias (int): Número de días para ejecutar la simulación.

        Retorna:
        Tuple[List[int], List[float], List[int]]: 
            - Lista con la cantidad de infectados cada turno.
            - Lista con la satisfacción agregada cada turno.
            - Lista con la cantidad de asistencias a todas las entidades cada turno.
        """
        historial_infectados = []
        historial_satisfaccion = []
        historial_asistencias = []

        for dia in range(dias):
            self.actualizar_riesgos()
            self.registrar_asistencias()
            infectados_reportados = sum(1 for estado in self.estados if (estado in 'I'))
            satisfaccion_agregada = 0
            asistencias = sum(self.asistencias_por_entidad.values())
            
            for i in range(self.n):
                for entidad in self.B.neighbors(i):
                    satisfaccion = self.B[i][entidad]['satisfaccion']
                    if self.individuo_asistio_a_entidad[i][entidad - self.n]:
                        satisfaccion_agregada += satisfaccion
            
            self.actualizar_estados_duracion()
            
            historial_infectados.append(infectados_reportados)
            historial_satisfaccion.append(satisfaccion_agregada)
            historial_asistencias.append(asistencias)
        
        return historial_infectados, historial_satisfaccion, historial_asistencias

def PlotSimulacion(
        titulo : str,
        historial_infectados : List[int],
        historial_satisfaccion : List[float],
        historial_asistencias : List[int],
):
    ax1.plot(historial_infectados, label='Infectados')
    ax1.plot(historial_satisfaccion, label='Satisfacción')
    asistencias_posibles = len(simulacion.asistencias_posibles())
    ax1.plot([x/asistencias_posibles * N for x in historial_asistencias], label='Asistencias')
    ax1.legend()

    m_infectados = max(historial_infectados)
    ax2.plot([x/m_infectados for x in historial_infectados], label = "Infectados reescalados (1 = maximo de infectados)")

    if max_tolerancia_riesgo>1000:
        ax1.set_title(f"Simulación con infinita tolerancia al riesgo")
        ax2.set_title(f"Simulación con infinita tolerancia al riesgo")
    else:
        ax1.set_title(f"Simulación con {max_tolerancia_riesgo:.4f} tolerancia al riesgo")
        ax2.set_title(f"Simulación con {max_tolerancia_riesgo:.4f} tolerancia al riesgo")
    
    ax2.legend()

    ax1.set_xlabel("Tiempo")
    ax2.set_xlabel("Tiempo")