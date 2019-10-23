import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


periodo = 2
desejada, prevista, acuracia = dict(), dict(), dict()
tp, fp, fn, tn, tpr, fpr = dict(), dict(), dict(), dict(), dict(), dict()
precision, sensibilidade, especificidade, acuracia = dict(), dict(), dict(), dict()


def get_rate(tpo, tne, fpo, fne):
    tprt = 100 * tpo / (tpo + fpo)
    fprt = 100 * tne / (tne + fne)
    return tprt, fprt


for per in range(1, periodo+1):
    for i in [0, 5, 10, 15, 20, 25, 30]:
        desejada[str(i)+'_'+str(per)] = np.genfromtxt('incremental/lbl_fluxos_'+str(i)+'min_'+str(per)+'.txt')
        prevista[str(i)+'_'+str(per)] = np.genfromtxt('incremental/outputs_'+str(i)+'min_'+str(per)+'.txt')
        acuracia[str(i)+'_'+str(per)] = accuracy_score(desejada[str(i)+'_'+str(per)], prevista[str(i)+'_'+str(per)])
        tp[str(i)+'_'+str(per)], fp[str(i)+'_'+str(per)], \
            fn[str(i)+'_'+str(per)], tn[str(i)+'_'+str(per)] = \
            confusion_matrix(desejada[str(i)+'_'+str(per)], prevista[str(i)+'_'+str(per)]).ravel()
        tpr[str(i)+'_'+str(per)], fpr[str(i)+'_'+str(per)] = \
            get_rate(tp[str(i)+'_'+str(per)], tn[str(i)+'_'+str(per)], fp[str(i)+'_'+str(per)], fn[str(i)+'_'+str(per)])
        precision[str(i)+'_'+str(per)] = (tp[str(i)+'_'+str(per)] / (tp[str(i)+'_'+str(per)] + fp[str(i)+'_'+str(per)])) * 100
        sensibilidade[str(i)+'_'+str(per)] = (tp[str(i)+'_'+str(per)] / (tp[str(i)+'_'+str(per)] + fn[str(i)+'_'+str(per)])) * 100
        especificidade[str(i)+'_'+str(per)] = (tn[str(i)+'_'+str(per)] / (tn[str(i)+'_'+str(per)] + fp[str(i)+'_'+str(per)])) * 100
        acuracia[str(i)+'_'+str(per)] = (accuracy_score(desejada[str(i)+'_'+str(per)], prevista[str(i)+'_'+str(per)])) * 100

fig, ax = plt.subplots()

x = ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30',
     '00:35', '00:40', '00:45', '00:50', '00:55', '01:00', '01:05']

ax.plot(x, list(precision.values()), color='red', label='Precision', marker='o')
ax.plot(x, list(sensibilidade.values()), color='yellow', label='Sensibility', marker='x')
ax.plot(x, list(especificidade.values()), color='blue', label='Specificity', marker='v')
ax.plot(x, list(acuracia.values()), color='green', label='Acuracy', marker='*')

# Tamanho dos Ticks
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# Labels
plt.xlabel("Tempo (min)", size=16)
plt.ylabel("Métrica (%)", size=16)
plt.ylim(90, 101)

plt.legend(loc='lower right', prop={'size': 16})
plt.show()
