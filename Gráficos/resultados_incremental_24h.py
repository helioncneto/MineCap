import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


periodo = 10
desejada, prevista, acuracia = dict(), dict(), dict()
tp, fp, fn, tn, tpr, fpr = dict(), dict(), dict(), dict(), dict(), dict()
precision, sensibilidade, especificidade, acuracia = dict(), dict(), dict(), dict()

def calc_media(lista):
    return sum(lista) / len(lista)


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

fig, ax = plt.subplots()
'''
x = ['00:00', '00:05', '00:10', '00:15', '00:20', '00:25', '00:30', '00:35', '00:40', '00:45', '00:50', '00:55',
'01:00', '01:05', '01:10', '01:15', '01:20', '01:25', '01:30', '01:35', '01:40', '01:45', '01:50', '01:55',
'02:00', '02:05', '02:10', '02:15', '02:20', '02:25', '02:30', '02:35', '02:40', '02:45', '02:50', '02:55',
'03:00', '03:05', '03:10', '03:15', '03:20', '03:25', '03:30', '03:35', '03:40', '03:45', '03:50', '03:55',
'04:00', '04:05', '04:10', '04:15', '04:20', '04:25', '04:30', '04:35', '04:40', '04:45', '04:50', '04:55',
'05:00', '05:05', '05:10', '05:15', '05:20', '05:25', '05:30', '05:35', '05:40', '05:45']
'''
precision_l = [precision['0_1']]
sensibilidade_l = [sensibilidade['0_1']]
especificidade_l = [especificidade['0_1']]
acuracia_l= [acuracia['0_1']]

for i in range(1, periodo+1):
    precision_l.append(calc_media([precision[str(5) + '_' + str(i)], precision[str(10) + '_' + str(i)],
                          precision[str(15) + '_' + str(i)], precision[str(20) + '_' + str(i)],
                          precision[str(25) + '_' + str(i)], precision[str(30) + '_' + str(i)]]))
    sensibilidade_l.append(calc_media([sensibilidade[str(5) + '_' + str(i)], sensibilidade[str(10) + '_' + str(i)],
                                       sensibilidade[str(15) + '_' + str(i)], sensibilidade[str(20) + '_' + str(i)],
                                       sensibilidade[str(25) + '_' + str(i)], sensibilidade[str(30) + '_' + str(i)]]))
    especificidade_l.append(calc_media([especificidade[str(5) + '_' + str(i)], especificidade[str(10) + '_' + str(i)],
                                        especificidade[str(15) + '_' + str(i)], especificidade[str(20) + '_' + str(i)],
                                        especificidade[str(25) + '_' + str(i)], especificidade[str(30) + '_' + str(i)]]))
    acuracia_l.append(calc_media([acuracia[str(5) + '_' + str(i)], acuracia[str(10) + '_' + str(i)],
                                  acuracia[str(15) + '_' + str(i)], acuracia[str(20) + '_' + str(i)],
                                  acuracia[str(25) + '_' + str(i)], acuracia[str(30) + '_' + str(i)]]))




x = ['00:00', '00:30', '01:00', '01:30', '02:00', '02:30', '03:00', '03:30', '04:00',
     '04:30', '05:00']
ax.plot(x, precision_l, color='red', label='Precision', marker='o')
ax.plot(x, sensibilidade_l, color='yellow', label='Sensibility', marker='x')
ax.plot(x, especificidade_l, color='blue', label='Specificity', marker='v')
ax.plot(x, acuracia_l, color='green', label='Acuracy', marker='*')
#ax.plot(x, list(acuracia.values()), color='green', label='Acuracy', marker='*')

# Tamanho dos Ticks
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# Labels
plt.xlabel("Tempo (min)", size=16)
plt.ylabel("Métrica (%)", size=16)
plt.ylim(90, 101)

plt.legend(loc='lower right', prop={'size': 16})
plt.show()