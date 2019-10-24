# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
import shutil
import os
import random
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
from soja_net import SojaNet
from logger import Logger
from config_manager import load_config

# Reproducibilidad (dentro de lo que permite la variabilidad en GPUs) ==
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
print("CUDA version", torch.version.cuda)
# =======================================================================

# Parametros globales ===================================================
config = load_config("config")
Nfolds = int(config["nfolds"])
data_dir = config["data_dir"]

nbatch = int(config["nbatch"])
device = config["device"]
output_dir = config["output_dir"]

shutil.rmtree(output_dir, ignore_errors=True)
os.mkdir(output_dir)
# ======================================================================

logger = Logger(output_dir)
logger.start("res")

criterion = torch.nn.MSELoss()

# Dataset
df = pd.read_csv(config["data_dir"] + "datasetRofex4.csv").set_index("Id")
df.Fecha = pd.to_datetime(df.Fecha, dayfirst=True)
df["mes"] = df.Fecha.apply(lambda x: x.month-1)
df["año"] = df.Fecha.apply(lambda x: x.year-2004)
df["diasemana"] = df.Fecha.apply(lambda x: x.weekday())
df = df.drop(columns=["Mon.", "Unidad"]) # valores constantes
categorical = [13, 14, 15]
numerical = len(df.columns) - 1 - len(categorical)

# parametros de normalización
m_cierre = 0
s_cierre = 1
df = df.fillna(0)
for f in range(len(df.columns)):
    if f not in categorical and df.columns[f] != "Fecha":
        m = df[df.columns[f]].mean()
        s = df[df.columns[f]].std()
        df[df.columns[f]] = (df[df.columns[f]]-m)/s
        if df.columns[f] == "Cierre":
            m_cierre = m
            s_cierre = s

df = df.fillna(0)

best_losses = []
logger.log("fold,it,train_loss,mse_optim,mse_test\n")
seqlength = 2000
best_outs = torch.zeros(Nfolds, 10).to(device) 
for fold in range(Nfolds):
    wait = 0
    best_loss = 9999
    best_it = 0 
    net = SojaNet([12, 13, 14], [12, 16, 7], numerical, device).to(device)
    
    for it in range(300):

        # generacion de puntos de evaluación aleatorios
        train_data = torch.zeros(nbatch, seqlength, len(df.columns) - 1).to(device)
        train_labels = torch.zeros(nbatch, 10).to(device)
        optim_data = torch.zeros(nbatch, seqlength+10, len(df.columns) - 1).to(device)
        optim_labels = torch.zeros(nbatch, 10).to(device)
        test_data = torch.zeros(nbatch, seqlength+10, len(df.columns) - 1).to(device)
        test_labels = torch.zeros(nbatch,  10).to(device)

        for b in range(nbatch):
            eval_point = np.random.randint(seqlength, len(df) - 20)
            train_data[b, :, :] = torch.tensor(df.iloc[eval_point-seqlength:eval_point].drop(columns=["Fecha"]).values).to(device)
            train_labels[b, :] = torch.tensor(df.iloc[eval_point:eval_point + 10].Cierre.values).to(device)

            optim_data[b, :, :] = torch.tensor(df.iloc[eval_point-seqlength:eval_point + 10].drop(columns=["Fecha"]).values).to(device)
            optim_labels[b, :] = torch.tensor(df.iloc[eval_point+10:eval_point + 20].Cierre.values).to(device)

            # test es siempre sobre el intervalo final
            test_data[b, :, :] = torch.tensor(df.iloc[-seqlength-20:-10].drop(columns=["Fecha"]).values).to(device)
            test_labels[b, :] = torch.tensor(df.iloc[-10:].Cierre.values).to(device)

        # Train
        net.train()
        net.optimizer.zero_grad()
        out = net(Variable(train_data))
        loss = criterion(out, Variable(train_labels))
        loss.backward()
        net.optimizer.step()
        loss_train = loss.item()
       
        # Test
        net.eval()
        out = net(Variable(optim_data)).detach()
        loss_optim = criterion(out, Variable(optim_labels)).item()

        net.eval()
        out = net(Variable(test_data)).detach()
        loss_test = criterion(out, Variable(test_labels)).item()
        logger.log("%d, %d, %.6f, %.6f, %.6f\n" % (fold, it, loss_train, loss_optim, loss_test))

        torch.cuda.empty_cache()

        if loss_test < best_loss:
            best_loss = loss_test
            best_out = out[0, :].detach()
            wait = 0
            best_it = it
        else:
            wait += 1
            if wait > 30:
                break

    print("best ", best_it, best_loss, best_out)
    best_outs[fold, :] = best_out
    best_losses.append(best_loss)
    with open(output_dir+"results_%d" %fold, "w") as fout:
        # Las 10 primeras lineas son los datos provistos (en principio no tenía sentido calcular nada para predecir...)
        for k in range(10):
            p1 = df.iloc[-10+k].Cierre * s_cierre + m_cierre
            fecha = df.iloc[-10+k].Fecha
            fecha = "%02d/%02d/%d" %(fecha.day, fecha.month, fecha.year)
            p0 = df.iloc[-10+k-1].Cierre  * s_cierre + m_cierre
            fout.write("%d, %s, %.6f, %.6f\n" %(k+1, fecha, (p1-p0)/p0, p1))
        p0 = p1
        # La prediccion de valores futuros se hace automáticamente a partir de las fechas hasta FC
        fechas = ["30/09/2019", "01/10/2019", "02/10/2019", "03/10/2019", "04/10/2019", "07/10/2019", "08/10/2019", "09/10/2019", "10/10/2019", "11/10/2019"]
        for k in range(10):
            p1 = best_out[k] * s_cierre + m_cierre
            fecha = fechas[k]
            fout.write("%d, %s, %.6f, %.6f\n" %(11+k, fecha, (p1-p0)/p0, p1))
            p0 = p1


# predicciones medias
print("average", np.mean(best_losses), np.std(best_losses))
best_outs_mean = torch.mean(best_outs, dim=0)
best_outs_median = torch.median(best_outs, dim=0).values
print(best_outs_mean.shape)
with open(output_dir+"results_all", "w") as fout:
    for k in range(10):
        p1 = df.iloc[-10+k].Cierre*s_cierre + m_cierre
        fecha = df.iloc[-10+k].Fecha
        fecha = "%02d/%02d/%d" %(fecha.day, fecha.month, fecha.year)
        p0 = df.iloc[-10+k-1].Cierre*s_cierre + m_cierre
        fout.write("%d, %s, %.6f, %.6f\n" % (k+1, fecha, (p1-p0)/p0, p1))
    p0 = p1
    for k in range(10):
        p1 = best_outs_mean[k]*s_cierre + m_cierre
        fecha = fechas[k]
        fout.write("%d, %s, %.6f, %.6f\n" % (11+k, fecha, (p1-p0)/p0, p1))
        p0 = p1
