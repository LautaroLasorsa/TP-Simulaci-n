import numpy as np
import random

class SimulacionAgrupada:

    def __init__(self, /, 
                 probs : list[float], 
                 etoi : float, 
                 itos : float ,  
                 prob_inicial : float, 
                 fun_miedo,
                 fun_contagio,
                 info_global : bool = True):
        """
        probs: La probabilidad de que una persona asista a cada establecimiento (independientes)
        prob_inicial: Proporción de la población inicialmente infectada (independiente de la pertenencia a grupos)
        nivel_miedo: Suceptibilidad de la gente al miedo por una infección total
        etoi: Probabilidad de que en un turno un expuesto pase a infectado (1/duracion_expuesto)
        itos: Probabilidad de que un infectado vuelva a ser suceptible (1/duracion_infectado)
        fun_contagio: fun_contagio(sanos_grupo, exp_grupo, asistencia), dados los sanos y los expuestos de ese grupo calcula la tasa de contagio
        fun_miedo: fun_miedo(casos), dados los casos de la población total estima la probabilidad de salir de la población
        info_global: Si la info que tienen es solo a nivel global o si en cada caso se mira la info de ese lugar
        """

        self.etoi = etoi
        self.itos = itos
        self.fun_miedo = fun_miedo
        self.fun_contagio = fun_contagio
        self.prob_lugar = probs
        self.n = len(probs)
        self.prob_grupos = [0] * (1<<self.n)
        self.infc_grupos = [ [(0,0)] for _ in range(1<<self.n) ] # (expuestos, infectados)
        self.infc_total = [(prob_inicial,0)]
        self.infc_lugar = [[(prob_inicial,0)] for _ in range(self.n)]
        self.exp_total_new = [prob_inicial]
        self.asistencias = [[1,1]] # (quieren asistir, asisten realmente)
        self.info_global = info_global

        for i in range(len(self.prob_grupos)):
            p = 1
            for j in range(len(probs)):
                if i & (1<<j): p *= probs[j]
                else : p *= 1-probs[j]
            self.prob_grupos[i] = p

        self.infc_grupos[random.randint(0,len(self.prob_grupos)-1)] = [(prob_inicial,0)]

    def Avanzar(self):
        """
            Avanza un turno en la simulación
        """

        exp_prev = [0] * self.n
        san_prev = [0] * self.n
        exp_new = 0
        exp_t = 0
        inf_t = 0


        for i in range(1<<self.n):
            for j in range(self.n):
                if i & (1<<j):
                    exp_prev[j] += self.prob_grupos[i] * self.infc_grupos[i][-1][0] * (1 - self.etoi)
                    san_prev[j] += self.prob_grupos[i] * (1 - self.infc_grupos[i][-1][0] - self.infc_grupos[i][-1][1] * (1-self.itos))
        
        for i in range(self.n):
            exp_prev[i] /= self.prob_lugar[i]
            san_prev[i] /= self.prob_lugar[i]

        if self.info_global :
            asistencia = [self.fun_miedo(self.infc_total)]*self.n
        else :
            asistencia = [self.fun_miedo(self.infc_lugar[i]) for i in range(self.n)]

        intencion_t  = 0
        asistencia_t = 0

        for i in range(self.n):
            self.infc_lugar[i].append([0,0])

        for i in range(1<<self.n):
            (exp, inf) = self.infc_grupos[i][-1]
            exp2, inf2 = exp*(1-self.etoi), inf*(1-self.itos)+exp*self.etoi
            sanos2 = 1 - exp2 - inf2
            popcount = 0
            for j in range(self.n): 
                if i & (1<<j): popcount += 1
                
            for j in range(self.n):
                if i & (1<<j):
                    loc_e = sanos2 * asistencia[j] * self.fun_contagio(san_prev[j], exp_prev[j], asistencia[j])
                    sanos2 -= loc_e
                    exp_new += loc_e * self.prob_grupos[i]
                    exp2 += loc_e
                    intencion_t += asistencia[j] * self.prob_grupos[i] / popcount
                    asistencia_t += asistencia[j] * self.prob_grupos[i] / popcount * (1-inf2)
                    
            self.infc_grupos[i].append((exp2, inf2))
            exp_t += self.prob_grupos[i] * exp2
            inf_t += self.prob_grupos[i] * inf2

            for j in range(self.n):
                if i & (1<<j):
                    peso = self.prob_grupos[i]/self.prob_lugar[j]
                    self.infc_lugar[j][-1][0] += peso * exp2
                    self.infc_lugar[j][-1][1] += peso * inf2

        self.infc_total.append((exp_t,inf_t))
        self.asistencias.append((intencion_t, asistencia_t))
        self.exp_total_new.append(exp_new)                    
        
    def AvanzarHastaT(self, t):
        while len(self.infc_total)<=t:
            self.Avanzar()
