# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
# Graficando algunos resultados una vez hechas las predicciones.

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Evolución del entrenamiento
plt.figure()
log = pd.read_csv("results/res.log", sep=",")
sns.lineplot(data=log, x="it", y="mse_optim", label="MSE optimización")
sns.lineplot(data=log, x="it", y="mse_test", label="MSE test")
plt.title("Evolución del MSE")
plt.xlabel("Iteraciones")
plt.ylabel("MSE")

plt.savefig("figs/MSEvsIter.svg")

# Resultados
plt.figure()
ref = pd.read_csv("data/futuros_finales.csv")
fechas = [s[:10] for s in ref.Fecha]
cierres = ref.Cierre.values
retornos = [0] + [(cierres[p]-cierres[p-1]) / cierres[p-1] for p in range(1, 20)]

res = pd.read_csv("results/results_all", sep=",", header=None).iloc[10:, 3].values

plt.plot(cierres, '*-', label="Referencia")
plt.plot(np.arange(10, 20), res, '*-', label="Predicción media")
res = np.zeros((10, 10))
for fold in range(10):
    res[fold, :] = pd.read_csv("results/results_%d" % fold, sep=",", header=None)[3].iloc[10:].values

res25 = np.percentile(res, 25, axis=0)
res75 = np.percentile(res, 75, axis=0)
plt.plot(np.arange(10, 20), res25, '--', color="blue", label="Percentil 25-75")
plt.plot(np.arange(10, 20), res75, '--', color="blue")

plt.xticks(np.arange(20), fechas, rotation=90)
plt.xlabel("Fechas")
plt.ylabel("Precio de la soja al cierre [USD por tonelada]")
plt.legend()
plt.title("Evolución de precios y predicciones a partir de la FC")
plt.tight_layout()

plt.plot([9, 9], [min(cierres), max(cierres)], '--', color="black")
plt.text(9.3, max(cierres), "FC")

plt.savefig("figs/results.svg")

# Grafica de retornos
plt.figure()
plt.plot(retornos, '*-', label="Referencia retornos")
res = pd.read_csv("results/results_all", sep=",", header=None).iloc[10:, 2].values
plt.plot(np.arange(10, 20), res, '*-', label="Predicción de retornos media")
res = np.zeros((10, 10))
for fold in range(10):
    res[fold, :] = pd.read_csv("results/results_%d" % fold, sep=",", header=None)[2].iloc[10:].values

res25 = np.percentile(res, 25, axis=0)
res75 = np.percentile(res, 75, axis=0)
plt.plot(np.arange(10, 20), res25, '--', color="blue", label="Percentil 25-75")
plt.plot(np.arange(10, 20), res75, '--', color="blue")
plt.xticks(np.arange(20), fechas, rotation=90)
plt.xlabel("Fechas")
plt.ylabel("Retorno simple de la soja al cierre")
plt.legend()
plt.title("Evolución del retorno simple y predicciones a partir de la FC")
plt.tight_layout()

plt.plot([9, 9], [min(retornos), max(retornos)], '--', color="black")
plt.text(9.3, max(retornos), "FC")

plt.savefig("figs/results_ret.svg")

plt.show()