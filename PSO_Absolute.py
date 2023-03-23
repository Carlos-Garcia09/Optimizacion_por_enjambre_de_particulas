import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import time

class Particula:
    def __init__(self, posicion, velocidad):
        self.posicion = np.array(posicion)
        self.velocidad = np.array(velocidad)
        self.mejorLocal = np.array(posicion)

R = np.array([[-100,100],[-100,100]])
def f(x):
    return abs(x[0]) + abs(x[1])
    
def f_graf(x,y):
    return abs(x) + abs(y)

toleranciaFitness = -0.5 # Tolerancia de fitness


n = 50          # Particulas
N = len(R)      # Dimensiones
E = 200         # Epocas
maxIteraciones = 200 # Maximo numero de iteraciones
enjambre = []
particulaGanadora = 0

print("Dimensión: ", N)

t = time.time()

# Inicializamos las particulas aleatoreamente
for k in range(n): # Cada particula
    # Para cada dimension
    posicion = []
    for i in range(N):
        posicion.append(random.uniform(R[i][0], R[i][1]))
    enjambre.append(Particula(posicion, [0]*N)) # Particula inicializada


mejorGlobal = enjambre[0].mejorLocal # Posicion inicial del mejor global
mejorFitness = f(mejorGlobal) # Fitness inicial del mejor global
fitnessEvolution = []
i = 0
while (mejorFitness > toleranciaFitness).all() and (i < maxIteraciones):
    # Para cada particula
    for p, particula in enumerate(enjambre):
        # Si la posicion actual del individuo es mejor que su mejor local, entonces actualizo
        if (f(particula.posicion) < f(particula.mejorLocal)).all():
            particula.mejorLocal = particula.posicion
        # Si la mejor posicion histórica del individuo es mejor que la posición global de la bandada, entonces actualizo
        if (f(particula.mejorLocal) < f(mejorGlobal)).all():
            mejorGlobal = particula.mejorLocal
            mejorFitness = f(mejorGlobal)
            particulaGanadora = p
    
    for p, particula in enumerate(enjambre):
        # Para cada particula

        r1 = np.random.rand(2)
        r2 = np.random.rand(2)


        c1 = 2.5 - (2 / maxIteraciones) * i
        c2 = .5 - (2 / maxIteraciones)  * i
        
        #actualizo velocidad
        particula.velocidad = particula.velocidad + c1 * r1 * (particula.mejorLocal - particula.posicion) + c2 * r2 *(mejorGlobal - particula.posicion)    
        #actualizo posicion
        particula.posicion = particula.posicion +  particula.velocidad
        print(particula.posicion)
    
    fitnessEvolution.append(mejorFitness)
    i += 1

print ("Valor Optimo: ", mejorFitness)
print ("Mejor solucion encontrada:", enjambre[particulaGanadora].posicion)

plt.plot(fitnessEvolution)
plt.title('Optimización PSO de la funcion absoluta')
plt.xlabel('Iteración')
plt.ylabel('Evolución de la función de desempeño')
plt.show()
