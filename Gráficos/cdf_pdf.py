import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter

arq = open('..\process-layer\DT\set_latency.txt', 'r')
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

fig, ax = plt.subplots()
#x = np.array(dic_percent.)
print(x)


ax.plot(x, cdf , markerfacecolor='skyblue', markersize=5)
ax.set_xticklabels(x, fontsize=12)
#ax.set_yticklabels(('0','1', '2', '3', '4', '5', '6'), fontsize=12)
#plt.margins(0.02)
plt.show()






