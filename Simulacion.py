import random
import math
import numpy as np

def Simular(n : int, m : int, / , 
            alfa : float, t : 
            int, seed : int = 20_05_2024, 
            dias : int = 7, diasBase = 5,
            lat : int = 5, latBase = 3,
            memory : float = 0.9,
            beta : float = 1,
            ) -> tuple[list[float],list[float], list[list[float]]]:
    """
    n personas,
    m actividaes
    alfa costo de % de contagios
    t # turnos
    seed: Semilla de la simulación
    dias y diasBase: La enfermedad dura diasBase + Binomial(dias,0.5) turnos
    lat y latBase: La latencia dura latBase + Binomial(lat,0.5) turnos
    memory: memoria de los riesgos
    beta: riesgo de contagio. Una persona expuesta a un lugar con A latentes y B asistentes sigue sana con probabilidad exp(-A/(beta*B))
    """
    random.seed(seed)
    np.random.seed(seed)

    contagios = [0] * t
    satisfaccion = [0] * t

    eps = 0.25

    adj = [
        [ random.lognormvariate(0,1) - eps for _ in range(m) ] for _ in range(n)
    ]


    # Regularizo la utilidad
    for i in range(n):
        #for j in range(m):
        #    if adj[i][j] <= 100: adj[i][j] = -1
        suma = sum([adj[i][j] for j in range(m) if adj[i][j] > 0])
        for j in range(m):
            adj[i][j] /= suma

    estado = [0] * n
    estado[random.randint(0,n-1)] = latBase + np.random.binomial(lat,0.5)
    #expuestos = [0] * m
    #asis = [1] * m
    riesgo = [0] * m

    for turno in range(t):
        visExp = [0.0] * m
        trueExp = [0.0] * m
        sigAsis   = [1.0] * m
        
        # print(riesgo)

        for i in range(n):
            if estado[i] >= 0:
                for j in range(m):
                    if riesgo[j] * alfa < adj[i][j]:
                        satisfaccion[turno] += adj[i][j]
                        sigAsis[j] += 1
                        visExp[j] += int( estado[i] == 1 )
                        trueExp[j] += int( estado[i] >= 1 )

        for i in range(n):
            if estado[i] == 0:
                probSano = 1
                for j in range(m):
                    if riesgo[j] * alfa < adj[i][j]:
                        probSano *= math.exp(-trueExp[j]/(beta *sigAsis[j]))
                if random.random() > probSano:
                    estado[i] = np.random.binomial(lat,0.5) + latBase
                    contagios[turno] += 1
            elif estado[i] == 1:
                estado[i] = - (np.random.binomial(dias,0.5) + diasBase)
            elif estado[i] > 1: estado[i] -= 1
            else : estado[i] += 1

        for j in range(m):
            riesgo[j] = memory * riesgo[j] + (1-memory) * (visExp[j]/sigAsis[j])

        #expuestos = visExp
        #asis = sigAsis    
    
    return contagios, satisfaccion, adj
