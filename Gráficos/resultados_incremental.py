import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

desejada_inc_0 = np.genfromtxt('incremental/lbl_fluxos.txt')
desejada_inc_5 = np.genfromtxt('incremental/lbl_fluxos_5min.txt')
desejada_inc_10 = np.genfromtxt('incremental/lbl_fluxos_10min.txt')
desejada_inc_15 = np.genfromtxt('incremental/lbl_fluxos_15min.txt')
desejada_inc_20 = np.genfromtxt('incremental/lbl_fluxos_20min.txt')
desejada_inc_25 = np.genfromtxt('incremental/lbl_fluxos_25min.txt')
desejada_inc_30 = np.genfromtxt('incremental/lbl_fluxos_30min.txt')

prevista_inc_0 = np.genfromtxt('incremental/outputs.txt')
prevista_inc_5 = np.genfromtxt('incremental/outputs_5min.txt')
prevista_inc_10 = np.genfromtxt('incremental/outputs_10min.txt')
prevista_inc_15 = np.genfromtxt('incremental/outputs_15min.txt')
prevista_inc_20 = np.genfromtxt('incremental/outputs_20min.txt')
prevista_inc_25 = np.genfromtxt('incremental/outputs_25min.txt')
prevista_inc_30 = np.genfromtxt('incremental/outputs_30min.txt')

# Acuracia
acuracia_inc_0 = accuracy_score(desejada_inc_0, prevista_inc_0)
acuracia_inc_5 = accuracy_score(desejada_inc_5, prevista_inc_5)
acuracia_inc_10 = accuracy_score(desejada_inc_10, prevista_inc_10)
acuracia_inc_15 = accuracy_score(desejada_inc_15, prevista_inc_15)
acuracia_inc_20 = accuracy_score(desejada_inc_20, prevista_inc_20)
acuracia_inc_25 = accuracy_score(desejada_inc_25, prevista_inc_25)
acuracia_inc_30 = accuracy_score(desejada_inc_30, prevista_inc_30)

tp_inc_0, fp_inc_0, fn_inc_0, tn_inc_0 = confusion_matrix(desejada_inc_0, prevista_inc_0).ravel()
tp_inc_5, fp_inc_5, fn_inc_5, tn_inc_5 = confusion_matrix(desejada_inc_5, prevista_inc_5).ravel()
tp_inc_10, fp_inc_10, fn_inc_10, tn_inc_10 = confusion_matrix(desejada_inc_10, prevista_inc_10).ravel()
tp_inc_15, fp_inc_15, fn_inc_15, tn_inc_15 = confusion_matrix(desejada_inc_15, prevista_inc_15).ravel()
tp_inc_20, fp_inc_20, fn_inc_20, tn_inc_20 = confusion_matrix(desejada_inc_20, prevista_inc_20).ravel()
tp_inc_25, fp_inc_25, fn_inc_25, tn_inc_25 = confusion_matrix(desejada_inc_25, prevista_inc_25).ravel()
tp_inc_30, fp_inc_30, fn_inc_30, tn_inc_30 = confusion_matrix(desejada_inc_30, prevista_inc_30).ravel()


def get_rate(tp, tn, fp, fn):
    tpr = 100 * tp / (tp + fp)
    fpr = 100 * tn / (tn + fn)

    return tpr, fpr


tpr_inc_0, fpr_inc_0 = get_rate(tp_inc_0, tn_inc_0, fp_inc_0, fn_inc_0)
tpr_inc_5, fpr_inc_5 = get_rate(tp_inc_5, tn_inc_5, fp_inc_5, fn_inc_5)
tpr_inc_10, fpr_inc_10 = get_rate(tp_inc_10, tn_inc_10, fp_inc_10, fn_inc_10)
tpr_inc_15, fpr_inc_15 = get_rate(tp_inc_15, tn_inc_15, fp_inc_15, fn_inc_15)
tpr_inc_20, fpr_inc_20 = get_rate(tp_inc_20, tn_inc_20, fp_inc_20, fn_inc_20)
tpr_inc_25, fpr_inc_25 = get_rate(tp_inc_25, tn_inc_25, fp_inc_25, fn_inc_25)
tpr_inc_30, fpr_inc_30 = get_rate(tp_inc_30, tn_inc_30, fp_inc_30, fn_inc_30)

precision_inc_0 = tp_inc_0 / (tp_inc_0 + fp_inc_0)
sensibilidade_inc_0 = tp_inc_0 / (tp_inc_0 + fn_inc_0)
especificidade_inc_0 = tn_inc_0 / (tn_inc_0 + fp_inc_0)

precision_inc_5 = tp_inc_5 / (tp_inc_5 + fp_inc_5)
sensibilidade_inc_5 = tp_inc_5 / (tp_inc_5 + fn_inc_5)
especificidade_inc_5 = tn_inc_5 / (tn_inc_5 + fp_inc_5)

precision_inc_10 = tp_inc_10 / (tp_inc_10 + fp_inc_10)
sensibilidade_inc_10 = tp_inc_10 / (tp_inc_10 + fn_inc_10)
especificidade_inc_10 = tn_inc_10 / (tn_inc_10 + fp_inc_10)

precision_inc_15 = tp_inc_15 / (tp_inc_15 + fp_inc_15)
sensibilidade_inc_15 = tp_inc_15 / (tp_inc_15 + fn_inc_15)
especificidade_inc_15 = tn_inc_15 / (tn_inc_15 + fp_inc_15)

precision_inc_20 = tp_inc_20 / (tp_inc_20 + fp_inc_20)
sensibilidade_inc_20 = tp_inc_20 / (tp_inc_20 + fn_inc_20)
especificidade_inc_20 = tn_inc_20 / (tn_inc_20 + fp_inc_20)

precision_inc_25 = tp_inc_25 / (tp_inc_25 + fp_inc_25)
sensibilidade_inc_25 = tp_inc_25 / (tp_inc_25 + fn_inc_25)
especificidade_inc_25 = tn_inc_25 / (tn_inc_25 + fp_inc_25)

precision_inc_30 = tp_inc_30 / (tp_inc_30 + fp_inc_30)
sensibilidade_inc_30 = tp_inc_30 / (tp_inc_30 + fn_inc_30)
especificidade_inc_30 = tn_inc_30 / (tn_inc_30 + fp_inc_30)

precision_inc_0 = precision_inc_0 * 100
sensibilidade_inc_0 = sensibilidade_inc_0 * 100
especificidade_inc_0 = especificidade_inc_0 * 100
acuracia_inc_0 = acuracia_inc_0 * 100

precision_inc_5 = precision_inc_5 * 100
sensibilidade_inc_5 = sensibilidade_inc_5 * 100
especificidade_inc_5 = especificidade_inc_5 * 100
acuracia_inc_5 = acuracia_inc_5 * 100

precision_inc_10 = precision_inc_10 * 100
sensibilidade_inc_10 = sensibilidade_inc_10 * 100
especificidade_inc_10 = especificidade_inc_10 * 100
acuracia_inc_10 = acuracia_inc_10 * 100

precision_inc_15 = precision_inc_15 * 100
sensibilidade_inc_15 = sensibilidade_inc_15 * 100
especificidade_inc_15 = especificidade_inc_15 * 100
acuracia_inc_15 = acuracia_inc_15 * 100

precision_inc_20 = precision_inc_20 * 100
sensibilidade_inc_20 = sensibilidade_inc_20 * 100
especificidade_inc_20 = especificidade_inc_20 * 100
acuracia_inc_20 = acuracia_inc_20 * 100

precision_inc_25 = precision_inc_25 * 100
sensibilidade_inc_25 = sensibilidade_inc_25 * 100
especificidade_inc_25 = especificidade_inc_25 * 100
acuracia_inc_25 = acuracia_inc_25 * 100

precision_inc_30 = precision_inc_30 * 100
sensibilidade_inc_30 = sensibilidade_inc_30 * 100
especificidade_inc_30 = especificidade_inc_30 * 100
acuracia_inc_30 = acuracia_inc_30 * 100



fig, ax = plt.subplots()

x = ['0', '5', '10', '15', '20', '25', '30']

prec_inc = [precision_inc_0, precision_inc_5, precision_inc_10, precision_inc_15,
            precision_inc_20, precision_inc_25, precision_inc_30]
ax.plot(x, prec_inc, color='red', label='Precisão', marker='o')
sens_inc = [sensibilidade_inc_0, sensibilidade_inc_5, sensibilidade_inc_10, sensibilidade_inc_15,
            sensibilidade_inc_20, sensibilidade_inc_25, sensibilidade_inc_30]
ax.plot(x, sens_inc, color='yellow', label='Sensibilidade', marker='x')
espe_inc = [especificidade_inc_0, especificidade_inc_5, especificidade_inc_10,
            especificidade_inc_15, especificidade_inc_20, especificidade_inc_25,
            especificidade_inc_30]
ax.plot(x, espe_inc, color='blue', label='Especificidade', marker='v')
acur_inc = [acuracia_inc_0, acuracia_inc_5, acuracia_inc_10, acuracia_inc_15,
            acuracia_inc_20, acuracia_inc_25, acuracia_inc_30]
ax.plot(x, acur_inc, color='green', label='Acurácia', marker='*')

# Tamanho dos Ticks
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

# Labels
plt.xlabel("Tempo (min)", size=16)
plt.ylabel("Métrica (%)", size=16)

plt.ylim(90, 101)

plt.legend(loc='lower right', prop={'size': 16})
plt.show()
