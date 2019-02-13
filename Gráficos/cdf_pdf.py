import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

def cdf(model):
    arq = open('../process-layer/' +model+'/set_latency.txt', 'r')
    texto = arq.readlines()

    latencias_str = []
    for linha in texto :
        linha = linha.split('\n')
        latencias_str.append(linha[0])
    arq.close()

    latencias = []
    for i in latencias_str:
        lat = float(i)
        latencias.append(round(lat, 1))

    dic_lat = {}

    for i in latencias:
        dic_lat[i] = latencias.count(i)
        #print("latencia %.1f : %d" % (i, latencias.count(i)))

    soma = 0
    for i in dic_lat:
        soma += dic_lat[i]
    dic_percent = {}
    lista_percent = []
    for i in dic_lat:
        dic_percent[i] = dic_lat[i]/soma

    dic_percent = sorted(dic_percent.items(), key=itemgetter(0))
    x = []
    for i in dic_percent:
        x.append(i[0])
        lista_percent.append(i[1])

    cdf = []
    #lista_percent = sorted(lista_percent)
    aa = 0
    for i in range(len(lista_percent)):
        if aa == 0:
            cdf.append(lista_percent[i])
            aa += lista_percent[i]
        else:
            aa += lista_percent[i]
            cdf.append(aa)
    lista_p_around = []
    for i in lista_percent:
        lista_p_around.append(round(i, 2))

    return x, cdf, lista_p_around


def plot_cdf(x, cdf):
    fig, ax = plt.subplots()
    #x = np.array(dic_percent.)

    for i in range(len(cdf)):
        ax.plot(x[i], cdf[i], markerfacecolor='skyblue', markersize=5)
    ax.set_xlim([0.2, 12])
    ax.set_ylim([0.2, 1])

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', alpha=0.2)
    ax.xaxis.grid(True, color='gray', linestyle='dashed', alpha=0.2)

    ax.set_xlabel("Latência (ms)", fontsize=14)
    ax.set_ylabel("Probabilidade", fontsize=14)
    plt.show()


def plot_pdf(x, lista_p_around):
    fig, ax = plt.subplots()

    ax.bar(x, lista_p_around, align='center', color='blue')
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', alpha=0.2)
    ax.xaxis.grid(True, color='gray', linestyle='dashed', alpha=0.2)

    ax.set_xlabel("Latência (ms)", fontsize=14)
    ax.set_ylabel("Probabilidade", fontsize=14)

    plt.show()

x1, cdf1, lista_p_around1 = cdf('DT')
x2, cdf2, lista_p_around2 = cdf('LR')
plot_cdf([x1,x2], [cdf1, cdf2])
plot_pdf(x1, lista_p_around1)
