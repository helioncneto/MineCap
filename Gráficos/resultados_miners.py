#
#
from proj import plot_roc_multi
import numpy as np
import matplotlib.pyplot as plt


desejada_fa_1 = np.genfromtxt('../process-layer/FA/1-miner/lbl_fluxos.txt')
desejada_fa_2 = np.genfromtxt('../process-layer/FA/2-miner/lbl_fluxos.txt')
desejada_fa_3 = np.genfromtxt('../process-layer/FA/3-miner/lbl_fluxos.txt')
desejada_fa_4 = np.genfromtxt('../process-layer/FA/4-miner/lbl_fluxos.txt')
desejada_fa_5 = np.genfromtxt('../process-layer/FA/5-miner/lbl_fluxos.txt')
desejada_fa_6 = np.genfromtxt('../process-layer/FA/6-miner/lbl_fluxos.txt')
desejada_fa_7 = np.genfromtxt('../process-layer/FA/7-miner/lbl_fluxos.txt')
desejada_fa_8 = np.genfromtxt('../process-layer/FA/8-miner/lbl_fluxos.txt')
desejada_fa_9 = np.genfromtxt('../process-layer/FA/9-miner/lbl_fluxos.txt')
desejada_fa_10 = np.genfromtxt('../process-layer/FA/10-miner/lbl_fluxos.txt')
desejada_fa_11 = np.genfromtxt('../process-layer/FA/11-miner/lbl_fluxos.txt')
desejada_fa_12 = np.genfromtxt('../process-layer/FA/12-miner/lbl_fluxos.txt')
desejada_fa_13 = np.genfromtxt('../process-layer/FA/13-miner/lbl_fluxos.txt')
desejada_fa_14 = np.genfromtxt('../process-layer/FA/14-miner/lbl_fluxos.txt')
desejada_fa_15 = np.genfromtxt('../process-layer/FA/15-miner/lbl_fluxos.txt')
desejada_fa_16 = np.genfromtxt('../process-layer/FA/16-miner/lbl_fluxos.txt')

desejada_gb_1 = np.genfromtxt('../process-layer/GB/1-miner/lbl_fluxos.txt')
desejada_gb_2 = np.genfromtxt('../process-layer/GB/2-miner/lbl_fluxos.txt')
desejada_gb_3 = np.genfromtxt('../process-layer/GB/3-miner/lbl_fluxos.txt')
desejada_gb_4 = np.genfromtxt('../process-layer/GB/4-miner/lbl_fluxos.txt')
desejada_gb_5 = np.genfromtxt('../process-layer/GB/5-miner/lbl_fluxos.txt')
desejada_gb_6 = np.genfromtxt('../process-layer/GB/6-miner/lbl_fluxos.txt')
desejada_gb_7 = np.genfromtxt('../process-layer/GB/7-miner/lbl_fluxos.txt')
desejada_gb_8 = np.genfromtxt('../process-layer/GB/8-miner/lbl_fluxos.txt')
desejada_gb_9 = np.genfromtxt('../process-layer/GB/9-miner/lbl_fluxos.txt')
desejada_gb_10 = np.genfromtxt('../process-layer/GB/10-miner/lbl_fluxos.txt')
desejada_gb_11 = np.genfromtxt('../process-layer/GB/11-miner/lbl_fluxos.txt')
desejada_gb_12 = np.genfromtxt('../process-layer/GB/12-miner/lbl_fluxos.txt')
desejada_gb_13 = np.genfromtxt('../process-layer/GB/13-miner/lbl_fluxos.txt')
desejada_gb_14 = np.genfromtxt('../process-layer/GB/14-miner/lbl_fluxos.txt')
desejada_gb_15 = np.genfromtxt('../process-layer/GB/15-miner/lbl_fluxos.txt')
desejada_gb_16 = np.genfromtxt('../process-layer/GB/16-miner/lbl_fluxos.txt')

# In[2]:


prevista_fa_1 = np.genfromtxt('../process-layer/FA/1-miner/outputs.txt')
prevista_fa_2 = np.genfromtxt('../process-layer/FA/2-miner/outputs.txt')
prevista_fa_3 = np.genfromtxt('../process-layer/FA/3-miner/outputs.txt')
prevista_fa_4 = np.genfromtxt('../process-layer/FA/4-miner/outputs.txt')
prevista_fa_5 = np.genfromtxt('../process-layer/FA/5-miner/outputs.txt')
prevista_fa_6 = np.genfromtxt('../process-layer/FA/6-miner/outputs.txt')
prevista_fa_7 = np.genfromtxt('../process-layer/FA/7-miner/outputs.txt')
prevista_fa_8 = np.genfromtxt('../process-layer/FA/8-miner/outputs.txt')
prevista_fa_9 = np.genfromtxt('../process-layer/FA/9-miner/outputs.txt')
prevista_fa_10 = np.genfromtxt('../process-layer/FA/10-miner/outputs.txt')
prevista_fa_11 = np.genfromtxt('../process-layer/FA/11-miner/outputs.txt')
prevista_fa_12 = np.genfromtxt('../process-layer/FA/12-miner/outputs.txt')
prevista_fa_13 = np.genfromtxt('../process-layer/FA/13-miner/outputs.txt')
prevista_fa_14 = np.genfromtxt('../process-layer/FA/14-miner/outputs.txt')
prevista_fa_15 = np.genfromtxt('../process-layer/FA/15-miner/outputs.txt')
prevista_fa_16 = np.genfromtxt('../process-layer/FA/16-miner/outputs.txt')

prevista_gb_1 = np.genfromtxt('../process-layer/GB/1-miner/outputs.txt')
prevista_gb_2 = np.genfromtxt('../process-layer/GB/2-miner/outputs.txt')
prevista_gb_3 = np.genfromtxt('../process-layer/GB/3-miner/outputs.txt')
prevista_gb_4 = np.genfromtxt('../process-layer/GB/4-miner/outputs.txt')
prevista_gb_5 = np.genfromtxt('../process-layer/GB/5-miner/outputs.txt')
prevista_gb_6 = np.genfromtxt('../process-layer/GB/6-miner/outputs.txt')
prevista_gb_7 = np.genfromtxt('../process-layer/GB/7-miner/outputs.txt')
prevista_gb_8 = np.genfromtxt('../process-layer/GB/8-miner/outputs.txt')
prevista_gb_9 = np.genfromtxt('../process-layer/GB/9-miner/outputs.txt')
prevista_gb_10 = np.genfromtxt('../process-layer/GB/10-miner/outputs.txt')
prevista_gb_11 = np.genfromtxt('../process-layer/GB/11-miner/outputs.txt')
prevista_gb_12 = np.genfromtxt('../process-layer/GB/12-miner/outputs.txt')
prevista_gb_13 = np.genfromtxt('../process-layer/GB/13-miner/outputs.txt')
prevista_gb_14 = np.genfromtxt('../process-layer/GB/14-miner/outputs.txt')
prevista_gb_15 = np.genfromtxt('../process-layer/GB/15-miner/outputs.txt')
prevista_gb_16 = np.genfromtxt('../process-layer/GB/16-miner/outputs.txt')

#Acuracia
from sklearn.metrics import accuracy_score
acuracia_fa_1 = accuracy_score(desejada_fa_1, prevista_fa_1)
acuracia_fa_2 = accuracy_score(desejada_fa_2, prevista_fa_2)
acuracia_fa_3 = accuracy_score(desejada_fa_3, prevista_fa_3)
acuracia_fa_4 = accuracy_score(desejada_fa_4, prevista_fa_4)
acuracia_fa_5 = accuracy_score(desejada_fa_5, prevista_fa_5)
acuracia_fa_6 = accuracy_score(desejada_fa_6, prevista_fa_6)
acuracia_fa_7 = accuracy_score(desejada_fa_7, prevista_fa_7)
acuracia_fa_8 = accuracy_score(desejada_fa_8, prevista_fa_8)
acuracia_fa_9 = accuracy_score(desejada_fa_9, prevista_fa_9)
acuracia_fa_10 = accuracy_score(desejada_fa_10, prevista_fa_10)
acuracia_fa_11 = accuracy_score(desejada_fa_11, prevista_fa_11)
acuracia_fa_12 = accuracy_score(desejada_fa_12, prevista_fa_12)
acuracia_fa_13 = accuracy_score(desejada_fa_13, prevista_fa_13)
acuracia_fa_14 = accuracy_score(desejada_fa_14, prevista_fa_14)
acuracia_fa_15 = accuracy_score(desejada_fa_15, prevista_fa_15)
acuracia_fa_16 = accuracy_score(desejada_fa_16, prevista_fa_16)

acuracia_gb_1 = accuracy_score(desejada_gb_1, prevista_gb_1)
acuracia_gb_2 = accuracy_score(desejada_gb_2, prevista_gb_2)
acuracia_gb_3 = accuracy_score(desejada_gb_3, prevista_gb_3)
acuracia_gb_4 = accuracy_score(desejada_gb_4, prevista_gb_4)
acuracia_gb_5 = accuracy_score(desejada_gb_5, prevista_gb_5)
acuracia_gb_6 = accuracy_score(desejada_gb_6, prevista_gb_6)
acuracia_gb_7 = accuracy_score(desejada_gb_7, prevista_gb_7)
acuracia_gb_8 = accuracy_score(desejada_gb_8, prevista_gb_8)
acuracia_gb_9 = accuracy_score(desejada_gb_9, prevista_gb_9)
acuracia_gb_10 = accuracy_score(desejada_gb_10, prevista_gb_10)
acuracia_gb_11 = accuracy_score(desejada_gb_11, prevista_gb_11)
acuracia_gb_12 = accuracy_score(desejada_gb_12, prevista_gb_12)
acuracia_gb_13 = accuracy_score(desejada_gb_13, prevista_gb_13)
acuracia_gb_14 = accuracy_score(desejada_gb_14, prevista_gb_14)
acuracia_gb_15 = accuracy_score(desejada_gb_15, prevista_gb_15)
acuracia_gb_16 = accuracy_score(desejada_gb_16, prevista_gb_16)

from sklearn.metrics import confusion_matrix
tp_fa_1, fp_fa_1, fn_fa_1, tn_fa_1 = confusion_matrix(desejada_fa_1, prevista_fa_1).ravel()
tp_fa_2, fp_fa_2, fn_fa_2, tn_fa_2 = confusion_matrix(desejada_fa_2, prevista_fa_2).ravel()
tp_fa_3, fp_fa_3, fn_fa_3, tn_fa_3 = confusion_matrix(desejada_fa_3, prevista_fa_3).ravel()
tp_fa_4, fp_fa_4, fn_fa_4, tn_fa_4 = confusion_matrix(desejada_fa_4, prevista_fa_4).ravel()
tp_fa_5, fp_fa_5, fn_fa_5, tn_fa_5 = confusion_matrix(desejada_fa_5, prevista_fa_5).ravel()
tp_fa_6, fp_fa_6, fn_fa_6, tn_fa_6 = confusion_matrix(desejada_fa_6, prevista_fa_6).ravel()
tp_fa_7, fp_fa_7, fn_fa_7, tn_fa_7 = confusion_matrix(desejada_fa_7, prevista_fa_7).ravel()
tp_fa_8, fp_fa_8, fn_fa_8, tn_fa_8 = confusion_matrix(desejada_fa_8, prevista_fa_8).ravel()
tp_fa_9, fp_fa_9, fn_fa_9, tn_fa_9 = confusion_matrix(desejada_fa_9, prevista_fa_9).ravel()
tp_fa_10, fp_fa_10, fn_fa_10, tn_fa_10 = confusion_matrix(desejada_fa_10, prevista_fa_10).ravel()
tp_fa_11, fp_fa_11, fn_fa_11, tn_fa_11 = confusion_matrix(desejada_fa_11, prevista_fa_11).ravel()
tp_fa_12, fp_fa_12, fn_fa_12, tn_fa_12 = confusion_matrix(desejada_fa_12, prevista_fa_12).ravel()
tp_fa_13, fp_fa_13, fn_fa_13, tn_fa_13 = confusion_matrix(desejada_fa_13, prevista_fa_13).ravel()
tp_fa_14, fp_fa_14, fn_fa_14, tn_fa_14 = confusion_matrix(desejada_fa_14, prevista_fa_14).ravel()
tp_fa_15, fp_fa_15, fn_fa_15, tn_fa_15 = confusion_matrix(desejada_fa_15, prevista_fa_15).ravel()
tp_fa_16, fp_fa_16, fn_fa_16, tn_fa_16 = confusion_matrix(desejada_fa_16, prevista_fa_16).ravel()

tp_gb_1, fp_gb_1, fn_gb_1, tn_gb_1 = confusion_matrix(desejada_gb_1, prevista_gb_1).ravel()
tp_gb_2, fp_gb_2, fn_gb_2, tn_gb_2 = confusion_matrix(desejada_gb_2, prevista_gb_2).ravel()
tp_gb_3, fp_gb_3, fn_gb_3, tn_gb_3 = confusion_matrix(desejada_gb_3, prevista_gb_3).ravel()
tp_gb_4, fp_gb_4, fn_gb_4, tn_gb_4 = confusion_matrix(desejada_gb_4, prevista_gb_4).ravel()
tp_gb_5, fp_gb_5, fn_gb_5, tn_gb_5 = confusion_matrix(desejada_gb_5, prevista_gb_5).ravel()
tp_gb_6, fp_gb_6, fn_gb_6, tn_gb_6 = confusion_matrix(desejada_gb_6, prevista_gb_6).ravel()
tp_gb_7, fp_gb_7, fn_gb_7, tn_gb_7 = confusion_matrix(desejada_gb_7, prevista_gb_7).ravel()
tp_gb_8, fp_gb_8, fn_gb_8, tn_gb_8 = confusion_matrix(desejada_gb_8, prevista_gb_8).ravel()
tp_gb_9, fp_gb_9, fn_gb_9, tn_gb_9 = confusion_matrix(desejada_gb_9, prevista_gb_9).ravel()
tp_gb_10, fp_gb_10, fn_gb_10, tn_gb_10 = confusion_matrix(desejada_gb_10, prevista_gb_10).ravel()
tp_gb_11, fp_gb_11, fn_gb_11, tn_gb_11 = confusion_matrix(desejada_gb_11, prevista_gb_11).ravel()
tp_gb_12, fp_gb_12, fn_gb_12, tn_gb_12 = confusion_matrix(desejada_gb_12, prevista_gb_12).ravel()
tp_gb_13, fp_gb_13, fn_gb_13, tn_gb_13 = confusion_matrix(desejada_gb_13, prevista_gb_13).ravel()
tp_gb_14, fp_gb_14, fn_gb_14, tn_gb_14 = confusion_matrix(desejada_gb_14, prevista_gb_14).ravel()
tp_gb_15, fp_gb_15, fn_gb_15, tn_gb_15 = confusion_matrix(desejada_gb_15, prevista_gb_15).ravel()
tp_gb_16, fp_gb_16, fn_gb_16, tn_gb_16 = confusion_matrix(desejada_gb_16, prevista_gb_16).ravel()

def get_rate(tp, tn, fp, fn):
    tpr = 100 * tp / (tp + fp)
    fpr = 100 * tn / (tn + fn)

    return tpr, fpr

tpr_fa_1, fpr_fa_1 = get_rate(tp_fa_1, tn_fa_1, fp_fa_1, fn_fa_1)
tpr_fa_2, fpr_fa_2 = get_rate(tp_fa_2, tn_fa_2, fp_fa_2, fn_fa_2)
tpr_fa_3, fpr_fa_3 = get_rate(tp_fa_3, tn_fa_3, fp_fa_3, fn_fa_3)
tpr_fa_4, fpr_fa_4 = get_rate(tp_fa_4, tn_fa_4, fp_fa_4, fn_fa_4)
tpr_fa_5, fpr_fa_5 = get_rate(tp_fa_5, tn_fa_5, fp_fa_5, fn_fa_5)
tpr_fa_6, fpr_fa_6 = get_rate(tp_fa_6, tn_fa_6, fp_fa_6, fn_fa_6)
tpr_fa_7, fpr_fa_7 = get_rate(tp_fa_7, tn_fa_7, fp_fa_7, fn_fa_7)
tpr_fa_8, fpr_fa_8 = get_rate(tp_fa_8, tn_fa_8, fp_fa_8, fn_fa_8)
tpr_fa_9, fpr_fa_9 = get_rate(tp_fa_9, tn_fa_9, fp_fa_9, fn_fa_9)
tpr_fa_10, fpr_fa_10 = get_rate(tp_fa_10, tn_fa_10, fp_fa_10, fn_fa_10)
tpr_fa_11, fpr_fa_11 = get_rate(tp_fa_11, tn_fa_11, fp_fa_11, fn_fa_11)
tpr_fa_12, fpr_fa_12 = get_rate(tp_fa_12, tn_fa_12, fp_fa_12, fn_fa_12)
tpr_fa_13, fpr_fa_13 = get_rate(tp_fa_13, tn_fa_13, fp_fa_13, fn_fa_13)
tpr_fa_14, fpr_fa_14 = get_rate(tp_fa_14, tn_fa_14, fp_fa_14, fn_fa_14)
tpr_fa_15, fpr_fa_15 = get_rate(tp_fa_15, tn_fa_15, fp_fa_15, fn_fa_15)
tpr_fa_16, fpr_fa_16 = get_rate(tp_fa_16, tn_fa_16, fp_fa_16, fn_fa_16)

tpr_gb_1, fpr_gb_1 = get_rate(tp_gb_1, tn_gb_1, fp_gb_1, fn_gb_1)
tpr_gb_2, fpr_gb_2 = get_rate(tp_gb_2, tn_gb_2, fp_gb_2, fn_gb_2)
tpr_gb_3, fpr_gb_3 = get_rate(tp_gb_3, tn_gb_3, fp_gb_3, fn_gb_3)
tpr_gb_4, fpr_gb_4 = get_rate(tp_gb_4, tn_gb_4, fp_gb_4, fn_gb_4)
tpr_gb_5, fpr_gb_5 = get_rate(tp_gb_5, tn_gb_5, fp_gb_5, fn_gb_5)
tpr_gb_6, fpr_gb_6 = get_rate(tp_gb_6, tn_gb_6, fp_gb_6, fn_gb_6)
tpr_gb_7, fpr_gb_7 = get_rate(tp_gb_7, tn_gb_7, fp_gb_7, fn_gb_7)
tpr_gb_8, fpr_gb_8 = get_rate(tp_gb_8, tn_gb_8, fp_gb_8, fn_gb_8)
tpr_gb_9, fpr_gb_9 = get_rate(tp_gb_9, tn_gb_9, fp_gb_9, fn_gb_9)
tpr_gb_10, fpr_gb_10 = get_rate(tp_gb_10, tn_gb_10, fp_gb_10, fn_gb_10)
tpr_gb_11, fpr_gb_11 = get_rate(tp_gb_11, tn_gb_11, fp_gb_11, fn_gb_11)
tpr_gb_12, fpr_gb_12 = get_rate(tp_gb_12, tn_gb_12, fp_gb_12, fn_gb_12)
tpr_gb_13, fpr_gb_13 = get_rate(tp_gb_13, tn_gb_13, fp_gb_13, fn_gb_13)
tpr_gb_14, fpr_gb_14 = get_rate(tp_gb_14, tn_gb_14, fp_gb_14, fn_gb_14)
tpr_gb_15, fpr_gb_15 = get_rate(tp_gb_15, tn_gb_15, fp_gb_15, fn_gb_15)
tpr_gb_16, fpr_gb_16 = get_rate(tp_gb_16, tn_gb_16, fp_gb_16, fn_gb_16)

# Métricas Floresta Aleatória
precision_fa_1 = tp_fa_1 / (tp_fa_1 + fp_fa_1)
sensibilidade_fa_1 = tp_fa_1 / (tp_fa_1 + fn_fa_1)
especificidade_fa_1 = tn_fa_1 / (tn_fa_1 + fp_fa_1)

precision_fa_2 = tp_fa_2 / (tp_fa_2 + fp_fa_2)
sensibilidade_fa_2 = tp_fa_2 / (tp_fa_2 + fn_fa_2)
especificidade_fa_2 = tn_fa_2 / (tn_fa_2 + fp_fa_2)

precision_fa_3 = tp_fa_3 / (tp_fa_3 + fp_fa_3)
sensibilidade_fa_3 = tp_fa_3 / (tp_fa_3 + fn_fa_3)
especificidade_fa_3 = tn_fa_3 / (tn_fa_3 + fp_fa_3)

precision_fa_4 = tp_fa_4 / (tp_fa_4 + fp_fa_4)
sensibilidade_fa_4 = tp_fa_4 / (tp_fa_4 + fn_fa_4)
especificidade_fa_4 = tn_fa_4 / (tn_fa_4 + fp_fa_4)

precision_fa_5 = tp_fa_5 / (tp_fa_5 + fp_fa_5)
sensibilidade_fa_5 = tp_fa_5 / (tp_fa_5 + fn_fa_5)
especificidade_fa_5 = tn_fa_5 / (tn_fa_5 + fp_fa_5)

precision_fa_6 = tp_fa_6 / (tp_fa_6 + fp_fa_6)
sensibilidade_fa_6 = tp_fa_6 / (tp_fa_6 + fn_fa_6)
especificidade_fa_6 = tn_fa_6 / (tn_fa_6 + fp_fa_6)

precision_fa_7 = tp_fa_7 / (tp_fa_7 + fp_fa_7)
sensibilidade_fa_7 = tp_fa_7 / (tp_fa_7 + fn_fa_7)
especificidade_fa_7 = tn_fa_7 / (tn_fa_7 + fp_fa_7)

precision_fa_8 = tp_fa_8 / (tp_fa_8 + fp_fa_8)
sensibilidade_fa_8 = tp_fa_8 / (tp_fa_8 + fn_fa_8)
especificidade_fa_8 = tn_fa_8 / (tn_fa_8 + fp_fa_8)

precision_fa_9 = tp_fa_9 / (tp_fa_9 + fp_fa_9)
sensibilidade_fa_9 = tp_fa_9 / (tp_fa_9 + fn_fa_9)
especificidade_fa_9 = tn_fa_9 / (tn_fa_9 + fp_fa_9)

precision_fa_10 = tp_fa_10 / (tp_fa_10 + fp_fa_10)
sensibilidade_fa_10 = tp_fa_10 / (tp_fa_10 + fn_fa_10)
especificidade_fa_10 = tn_fa_10 / (tn_fa_10 + fp_fa_10)

precision_fa_11 = tp_fa_11 / (tp_fa_11 + fp_fa_11)
sensibilidade_fa_11 = tp_fa_11 / (tp_fa_11 + fn_fa_11)
especificidade_fa_11 = tn_fa_11 / (tn_fa_11 + fp_fa_11)

precision_fa_12 = tp_fa_12 / (tp_fa_12 + fp_fa_12)
sensibilidade_fa_12 = tp_fa_12 / (tp_fa_12 + fn_fa_12)
especificidade_fa_12 = tn_fa_12 / (tn_fa_12 + fp_fa_12)

precision_fa_13 = tp_fa_13 / (tp_fa_13 + fp_fa_13)
sensibilidade_fa_13 = tp_fa_13 / (tp_fa_13 + fn_fa_13)
especificidade_fa_13 = tn_fa_13 / (tn_fa_13 + fp_fa_13)

precision_fa_14 = tp_fa_14 / (tp_fa_14 + fp_fa_14)
sensibilidade_fa_14 = tp_fa_14 / (tp_fa_14 + fn_fa_14)
especificidade_fa_14 = tn_fa_14 / (tn_fa_14 + fp_fa_14)

precision_fa_15 = tp_fa_15 / (tp_fa_15 + fp_fa_15)
sensibilidade_fa_15 = tp_fa_15 / (tp_fa_15 + fn_fa_15)
especificidade_fa_15 = tn_fa_15 / (tn_fa_15 + fp_fa_15)

precision_fa_16 = tp_fa_16 / (tp_fa_16 + fp_fa_16)
sensibilidade_fa_16 = tp_fa_16 / (tp_fa_16 + fn_fa_16)
especificidade_fa_16 = tn_fa_16 / (tn_fa_16 + fp_fa_16)

#Métricas Gradient Booster Tree
precision_gb_1 = tp_gb_1 / (tp_gb_1 + fp_gb_1)
sensibilidade_gb_1 = tp_gb_1 / (tp_gb_1 + fn_gb_1)
especificidade_gb_1 = tn_gb_1 / (tn_gb_1 + fp_gb_1)

precision_gb_2 = tp_gb_2 / (tp_gb_2 + fp_gb_2)
sensibilidade_gb_2 = tp_gb_2 / (tp_gb_2 + fn_gb_2)
especificidade_gb_2 = tn_gb_2 / (tn_gb_2 + fp_gb_2)

precision_gb_3 = tp_gb_3 / (tp_gb_3 + fp_gb_3)
sensibilidade_gb_3 = tp_gb_3 / (tp_gb_3 + fn_gb_3)
especificidade_gb_3 = tn_gb_3 / (tn_gb_3 + fp_gb_3)

precision_gb_4 = tp_gb_4 / (tp_gb_4 + fp_gb_4)
sensibilidade_gb_4 = tp_gb_4 / (tp_gb_4 + fn_gb_4)
especificidade_gb_4 = tn_gb_4 / (tn_gb_4 + fp_gb_4)

precision_gb_5 = tp_gb_5 / (tp_gb_5 + fp_gb_5)
sensibilidade_gb_5 = tp_gb_5 / (tp_gb_5 + fn_gb_5)
especificidade_gb_5 = tn_gb_5 / (tn_gb_5 + fp_gb_5)

precision_gb_6 = tp_gb_6 / (tp_gb_6 + fp_gb_6)
sensibilidade_gb_6 = tp_gb_6 / (tp_gb_6 + fn_gb_6)
especificidade_gb_6 = tn_gb_6 / (tn_gb_6 + fp_gb_6)

precision_gb_7 = tp_gb_7 / (tp_gb_7 + fp_gb_7)
sensibilidade_gb_7 = tp_gb_7 / (tp_gb_7 + fn_gb_7)
especificidade_gb_7 = tn_gb_7 / (tn_gb_7 + fp_gb_7)

precision_gb_8 = tp_gb_8 / (tp_gb_8 + fp_gb_8)
sensibilidade_gb_8 = tp_gb_8 / (tp_gb_8 + fn_gb_8)
especificidade_gb_8 = tn_gb_8 / (tn_gb_8 + fp_gb_8)

precision_gb_9 = tp_gb_9 / (tp_gb_9 + fp_gb_9)
sensibilidade_gb_9 = tp_gb_9 / (tp_gb_9 + fn_gb_9)
especificidade_gb_9 = tn_gb_9 / (tn_gb_9 + fp_gb_9)

precision_gb_10 = tp_gb_10 / (tp_gb_10 + fp_gb_10)
sensibilidade_gb_10 = tp_gb_10 / (tp_gb_10 + fn_gb_10)
especificidade_gb_10 = tn_gb_10 / (tn_gb_10 + fp_gb_10)

precision_gb_11 = tp_gb_11 / (tp_gb_11 + fp_gb_11)
sensibilidade_gb_11 = tp_gb_11 / (tp_gb_11 + fn_gb_11)
especificidade_gb_11 = tn_gb_11 / (tn_gb_11 + fp_gb_11)

precision_gb_12 = tp_gb_12 / (tp_gb_12 + fp_gb_12)
sensibilidade_gb_12 = tp_gb_12 / (tp_gb_12 + fn_gb_12)
especificidade_gb_12 = tn_gb_12 / (tn_gb_12 + fp_gb_12)

precision_gb_13 = tp_gb_13 / (tp_gb_13 + fp_gb_13)
sensibilidade_gb_13 = tp_gb_13 / (tp_gb_13 + fn_gb_13)
especificidade_gb_13 = tn_gb_13 / (tn_gb_13 + fp_gb_13)

precision_gb_14 = tp_gb_14 / (tp_gb_14 + fp_gb_14)
sensibilidade_gb_14 = tp_gb_14 / (tp_gb_14 + fn_gb_14)
especificidade_gb_14 = tn_gb_14 / (tn_gb_14 + fp_gb_14)

precision_gb_15 = tp_gb_15 / (tp_gb_15 + fp_gb_15)
sensibilidade_gb_15 = tp_gb_15 / (tp_gb_15 + fn_gb_15)
especificidade_gb_15 = tn_gb_15 / (tn_gb_15 + fp_gb_15)
print(precision_gb_9)
print(sensibilidade_gb_9)
print(especificidade_gb_9)

precision_gb_16 = tp_gb_16 / (tp_gb_16 + fp_gb_16)
sensibilidade_gb_16 = tp_gb_16 / (tp_gb_16 + fn_gb_16)
especificidade_gb_16 = tn_gb_16 / (tn_gb_16 + fp_gb_16)

#print("Precisão RF 1 miner: ", precision_fa_1)
#print("Sensibilidade RF 1 miner: ", sensibilidade_fa_1)
#print("Especificidade RF 1 miner: ", especificidade_fa_1)
#print("Especificidade RF 1 miner: ", acuracia_fa_1)

#print("Precisão RF 2 miner: ", precision_fa_2)
#print("Sensibilidade RF 2 miner: ", sensibilidade_fa_2)
#print("Especificidade RF 2 miner: ", especificidade_fa_2)
#print("Especificidade RF 2 miner: ", acuracia_fa_2)

#proba_fa_1 = np.genfromtxt('../process-layer/FA/1-miner/proba.txt', delimiter=',')
#proba_fa_2 = np.genfromtxt('../process-layer/FA/2-miner/proba.txt', delimiter=',')

precision_fa_1 = precision_fa_1 * 100
precision_fa_2 = precision_fa_2 * 100
precision_fa_3 = precision_fa_3 * 100
precision_fa_4 = precision_fa_4 * 100
precision_fa_5 = precision_fa_5 * 100
precision_fa_6 = precision_fa_6 * 100
precision_fa_7 = precision_fa_7 * 100
precision_fa_8 = precision_fa_8 * 100
precision_fa_9 = precision_fa_9 * 100
precision_fa_10 = precision_fa_10 * 100
precision_fa_11 = precision_fa_11 * 100
precision_fa_12 = precision_fa_12 * 100
precision_fa_13 = precision_fa_13 * 100
precision_fa_14 = precision_fa_14 * 100
precision_fa_15 = precision_fa_15 * 100
precision_fa_16 = precision_fa_16 * 100

sensibilidade_fa_1 = sensibilidade_fa_1 * 100
sensibilidade_fa_2 = sensibilidade_fa_2 * 100
sensibilidade_fa_3 = sensibilidade_fa_3 * 100
sensibilidade_fa_4 = sensibilidade_fa_4 * 100
sensibilidade_fa_5 = sensibilidade_fa_5 * 100
sensibilidade_fa_6 = sensibilidade_fa_6 * 100
sensibilidade_fa_7 = sensibilidade_fa_7 * 100
sensibilidade_fa_8 = sensibilidade_fa_8 * 100
sensibilidade_fa_9 = sensibilidade_fa_9 * 100
sensibilidade_fa_10 = sensibilidade_fa_10 * 100
sensibilidade_fa_11 = sensibilidade_fa_11 * 100
sensibilidade_fa_12 = sensibilidade_fa_12 * 100
sensibilidade_fa_13 = sensibilidade_fa_13 * 100
sensibilidade_fa_14 = sensibilidade_fa_14 * 100
sensibilidade_fa_15 = sensibilidade_fa_15 * 100
sensibilidade_fa_16 = sensibilidade_fa_16 * 100

especificidade_fa_1 = especificidade_fa_1 * 100
especificidade_fa_2 = especificidade_fa_2 * 100
especificidade_fa_3 = especificidade_fa_3 * 100
especificidade_fa_4 = especificidade_fa_4 * 100
especificidade_fa_5 = especificidade_fa_5 * 100
especificidade_fa_6 = especificidade_fa_6 * 100
especificidade_fa_7 = especificidade_fa_7 * 100
especificidade_fa_8 = especificidade_fa_8 * 100
especificidade_fa_9 = especificidade_fa_9 * 100
especificidade_fa_10 = especificidade_fa_10 * 100
especificidade_fa_11 = especificidade_fa_11 * 100
especificidade_fa_12 = especificidade_fa_12 * 100
especificidade_fa_13 = especificidade_fa_13 * 100
especificidade_fa_14 = especificidade_fa_14 * 100
especificidade_fa_15 = especificidade_fa_15 * 100
especificidade_fa_16 = especificidade_fa_16 * 100

acuracia_fa_1 = acuracia_fa_1 * 100
acuracia_fa_2 = acuracia_fa_2 * 100
acuracia_fa_3 = acuracia_fa_3 * 100
acuracia_fa_4 = acuracia_fa_4 * 100
acuracia_fa_5 = acuracia_fa_5 * 100
acuracia_fa_6 = acuracia_fa_6 * 100
acuracia_fa_7 = acuracia_fa_7 * 100
acuracia_fa_8 = acuracia_fa_8 * 100
acuracia_fa_9 = acuracia_fa_9 * 100
acuracia_fa_10 = acuracia_fa_10 * 100
acuracia_fa_11 = acuracia_fa_11 * 100
acuracia_fa_12 = acuracia_fa_12 * 100
acuracia_fa_13 = acuracia_fa_13 * 100
acuracia_fa_14 = acuracia_fa_14 * 100
acuracia_fa_15 = acuracia_fa_15 * 100
acuracia_fa_16 = acuracia_fa_16 * 100

precision_gb_1 = precision_gb_1 * 100
precision_gb_2 = precision_gb_2 * 100
precision_gb_3 = precision_gb_3 * 100
precision_gb_4 = precision_gb_4 * 100
precision_gb_5 = precision_gb_5 * 100
precision_gb_6 = precision_gb_6 * 100
precision_gb_7 = precision_gb_7 * 100
precision_gb_8 = precision_gb_8 * 100
precision_gb_9 = precision_gb_9 * 100
precision_gb_10 = precision_gb_10 * 100
precision_gb_11 = precision_gb_11 * 100
precision_gb_12 = precision_gb_12 * 100
precision_gb_13 = precision_gb_13 * 100
precision_gb_14 = precision_gb_14 * 100
precision_gb_15 = precision_gb_15 * 100
precision_gb_16 = precision_gb_16 * 100

sensibilidade_gb_1 = sensibilidade_gb_1 * 100
sensibilidade_gb_2 = sensibilidade_gb_2 * 100
sensibilidade_gb_3 = sensibilidade_gb_3 * 100
sensibilidade_gb_4 = sensibilidade_gb_4 * 100
sensibilidade_gb_5 = sensibilidade_gb_5 * 100
sensibilidade_gb_6 = sensibilidade_gb_6 * 100
sensibilidade_gb_7 = sensibilidade_gb_7 * 100
sensibilidade_gb_8 = sensibilidade_gb_8 * 100
sensibilidade_gb_9 = sensibilidade_gb_9 * 100
sensibilidade_gb_10 = sensibilidade_gb_10 * 100
sensibilidade_gb_11 = sensibilidade_gb_11 * 100
sensibilidade_gb_12 = sensibilidade_gb_12 * 100
sensibilidade_gb_13 = sensibilidade_gb_13 * 100
sensibilidade_gb_14 = sensibilidade_gb_14 * 100
sensibilidade_gb_15 = sensibilidade_gb_15 * 100
sensibilidade_gb_16 = sensibilidade_gb_16 * 100

especificidade_gb_1 = especificidade_gb_1 * 100
especificidade_gb_2 = especificidade_gb_2 * 100
especificidade_gb_3 = especificidade_gb_3 * 100
especificidade_gb_4 = especificidade_gb_4 * 100
especificidade_gb_5 = especificidade_gb_5 * 100
especificidade_gb_6 = especificidade_gb_6 * 100
especificidade_gb_7 = especificidade_gb_7 * 100
especificidade_gb_8 = especificidade_gb_8 * 100
especificidade_gb_9 = especificidade_gb_9 * 100
especificidade_gb_10 = especificidade_gb_10 * 100
especificidade_gb_11 = especificidade_gb_11 * 100
especificidade_gb_12 = especificidade_gb_12 * 100
especificidade_gb_13 = especificidade_gb_13 * 100
especificidade_gb_14 = especificidade_gb_14 * 100
especificidade_gb_15 = especificidade_gb_15 * 100
especificidade_gb_16 = especificidade_gb_16 * 100

acuracia_gb_1 = acuracia_gb_1 * 100
acuracia_gb_2 = acuracia_gb_2 * 100
acuracia_gb_3 = acuracia_gb_3 * 100
acuracia_gb_4 = acuracia_gb_4 * 100
acuracia_gb_5 = acuracia_gb_5 * 100
acuracia_gb_6 = acuracia_gb_6 * 100
acuracia_gb_7 = acuracia_gb_7 * 100
acuracia_gb_8 = acuracia_gb_8 * 100
acuracia_gb_9 = acuracia_gb_9 * 100
acuracia_gb_10 = acuracia_gb_10 * 100
acuracia_gb_11 = acuracia_gb_11 * 100
acuracia_gb_12 = acuracia_gb_12 * 100
acuracia_gb_13 = acuracia_gb_13 * 100
acuracia_gb_14 = acuracia_gb_14 * 100
acuracia_gb_15 = acuracia_gb_15 * 100
acuracia_gb_16 = acuracia_gb_16 * 100

fig, ax = plt.subplots()

x = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
prec_fa = [precision_fa_1, precision_fa_2, precision_fa_3, precision_fa_4,
        precision_fa_5, precision_fa_6, precision_fa_7, precision_fa_8,
        precision_fa_9, precision_fa_10, precision_fa_11, precision_fa_12,
        precision_fa_13, precision_fa_14, precision_fa_15, precision_fa_16]
ax.plot(x, prec_fa, color='red', label='Precisão', marker='o')
sens_fa = [sensibilidade_fa_1, sensibilidade_fa_2, sensibilidade_fa_3, sensibilidade_fa_4,
        sensibilidade_fa_5, sensibilidade_fa_6, sensibilidade_fa_7, sensibilidade_fa_8,
        sensibilidade_fa_9, sensibilidade_fa_10, sensibilidade_fa_11, sensibilidade_fa_12,
        sensibilidade_fa_13, sensibilidade_fa_14, sensibilidade_fa_15, sensibilidade_fa_16]
ax.plot(x, sens_fa, color='yellow', label='Sensibilidade', marker='x')
espe_fa = [especificidade_fa_1, especificidade_fa_2, especificidade_fa_3,
        especificidade_fa_4, especificidade_fa_5, especificidade_fa_6,
        especificidade_fa_7, especificidade_fa_8, especificidade_fa_9,
        especificidade_fa_10, especificidade_fa_11, especificidade_fa_12,
        especificidade_fa_13, especificidade_fa_14, especificidade_fa_15, especificidade_fa_16]
ax.plot(x, espe_fa, color='blue', label='Especificidade', marker='v')
acur_fa = [acuracia_fa_1, acuracia_fa_2, acuracia_fa_3, acuracia_fa_4,
        acuracia_fa_5, acuracia_fa_6, acuracia_fa_7,
        acuracia_fa_8, acuracia_fa_9, acuracia_fa_10,
        acuracia_fa_11, acuracia_fa_12, acuracia_fa_13,
        acuracia_fa_14, acuracia_fa_15, acuracia_fa_16]
ax.plot(x, acur_fa, color='green', label='Acurácia', marker='*')

#Tamanho dos Ticks
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa (%)", size=16)

plt.ylim(0, 100)

plt.legend(loc='lower right', prop={'size': 16})
plt.show()

fig, ax = plt.subplots()
prec_gb = [precision_gb_1, precision_gb_2, precision_gb_3, precision_gb_4,
        precision_gb_5, precision_gb_6, precision_gb_7, precision_gb_8,
        precision_gb_9, precision_gb_10, precision_gb_11, precision_gb_12,
        precision_gb_13, precision_gb_14, precision_gb_15, precision_gb_16]
ax.plot(x, prec_gb, color='red', label='Precisão', marker='o')
sens_gb = [sensibilidade_gb_1, sensibilidade_gb_2, sensibilidade_gb_3, sensibilidade_gb_4,
        sensibilidade_gb_5, sensibilidade_gb_6, sensibilidade_gb_7, sensibilidade_gb_8,
        sensibilidade_gb_9, sensibilidade_gb_10, sensibilidade_gb_11, sensibilidade_gb_12,
        sensibilidade_gb_13, sensibilidade_gb_14, sensibilidade_gb_15, sensibilidade_gb_16]
ax.plot(x, sens_gb, color='yellow', label='Sensibilidade', marker='x')
espe_gb = [especificidade_gb_1, especificidade_gb_2, especificidade_gb_3,
        especificidade_gb_4, especificidade_gb_5, especificidade_gb_6,
        especificidade_gb_7, especificidade_gb_8, especificidade_gb_9,
        especificidade_gb_10, especificidade_gb_11, especificidade_gb_12,
        especificidade_gb_13, especificidade_gb_14, especificidade_gb_15, especificidade_gb_16]
ax.plot(x, espe_gb, color='blue', label='Especificidade', marker='v')
acur_gb = [acuracia_gb_1, acuracia_gb_2, acuracia_gb_3, acuracia_gb_4,
        acuracia_gb_5, acuracia_gb_6, acuracia_gb_7,
        acuracia_gb_8, acuracia_gb_9, acuracia_gb_10,
        acuracia_gb_11, acuracia_gb_12, acuracia_gb_13,
        acuracia_gb_14, acuracia_gb_15, acuracia_gb_16]
ax.plot(x, acur_gb, color='green', label='Acurácia', marker='*')

#Tamanho dos Ticks
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa (%)", size=16)

plt.ylim(0, 100)

plt.legend(loc='lower right', prop={'size': 16})
plt.show()



fig, ax = plt.subplots()

#fp = [0, fp_fa_1, fp_fa_2, fp_fa_3, fp_fa_4, fp_fa_5, fp_fa_6,
#      fp_fa_7, fp_fa_8, fp_fa_9, fp_fa_10, fp_fa_11, fp_fa_12,
#      fp_fa_13, fp_fa_14, fp_fa_15, fp_fa_16]
fpr_l_fa = [fpr_fa_1, fpr_fa_2, fpr_fa_3, fpr_fa_4, fpr_fa_5, fpr_fa_6,
      fpr_fa_7, fpr_fa_8, fpr_fa_9, fpr_fa_10, fpr_fa_11, fpr_fa_12,
      fpr_fa_13, fpr_fa_14, fpr_fa_15, fpr_fa_16]
ax.plot(x, fpr_l_fa, color='green', label='Taxa de Falsos Positivos (FA)')

#fn = [0, fn_fa_1, fn_fa_2, fn_fa_3, fn_fa_4, fn_fa_5, fn_fa_6,
#      fn_fa_7, fn_fa_8, fn_fa_9, fn_fa_10, fn_fa_11, fn_fa_12,
#      fn_fa_13, fn_fa_14, fn_fa_15, fn_fa_16]
#fpr = [0, fpr_fa_1, fpr_fa_2, fpr_fa_3, fpr_fa_4, fpr_fa_5, fpr_fa_6,
#      fpr_fa_7, fpr_fa_8, fpr_fa_9, fpr_fa_10, fpr_fa_11, fpr_fa_12,
#      fpr_fa_13, fpr_fa_14, fpr_fa_15, fpr_fa_16]
#ax.plot(x, fn, color='red', label='Falso Negativo')

#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa de Falsos Positivos (%)", size=16)

#print(confusion_matrix(desejada_fa_8, prevista_fa_8))

plt.legend(loc='upper right', prop={'size': 16})
plt.show()

fig, ax = plt.subplots()

#tp = [0, tp_fa_1, tp_fa_2, tp_fa_3, tp_fa_4, tp_fa_5, tp_fa_6,
#      tp_fa_7, tp_fa_8, tp_fa_9, tp_fa_10, tp_fa_11, tp_fa_12,
#      tp_fa_13, tp_fa_14, tp_fa_15, tp_fa_16]
#ax.plot(x, tp, color='green', label='Verdadeiro Positivo')

#tn = [0, tn_fa_1, tn_fa_2, tn_fa_3, tn_fa_4, tn_fa_5, tn_fa_6,
#      tn_fa_7, tn_fa_8, tn_fa_9, tn_fa_10, tn_fa_11, tn_fa_12,
#      tn_fa_13, tn_fa_14, tn_fa_15, tn_fa_16]
tpr_l_fa = [tpr_fa_1, tpr_fa_2, tpr_fa_3, tpr_fa_4, tpr_fa_5, tpr_fa_6,
      tpr_fa_7, tpr_fa_8, tpr_fa_9, tpr_fa_10, tpr_fa_11, tpr_fa_12,
      tpr_fa_13, tpr_fa_14, tpr_fa_15, tpr_fa_16]
ax.plot(x, tpr_l_fa, color='red', label='Taxa Verdadeiros Positivos (FA)')

#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa de Verdadeiros Positivos (%)", size=16)


plt.legend(loc='upper right', prop={'size': 16})
plt.show()

####################
fig, ax = plt.subplots()

fpr_l_gb = [fpr_gb_1, fpr_gb_2, fpr_gb_3, fpr_gb_4, fpr_gb_5, fpr_gb_6,
      fpr_gb_7, fpr_gb_8, fpr_gb_9, fpr_gb_10, fpr_gb_11, fpr_gb_12,
      fpr_gb_13, fpr_gb_14, fpr_gb_15, fpr_gb_16]
ax.plot(x, fpr_l_gb, color='green', label='Taxa de Falsos Positivos (GB)')


#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa de Falsos Positivos (%)", size=16)


plt.legend(loc='upper right', prop={'size': 16})
plt.show()

fig, ax = plt.subplots()

tpr_l_gb = [tpr_gb_1, tpr_gb_2, tpr_gb_3, tpr_gb_4, tpr_gb_5, tpr_gb_6,
      tpr_gb_7, tpr_gb_8, tpr_gb_9, tpr_gb_10, tpr_gb_11, tpr_gb_12,
      tpr_gb_13, tpr_gb_14, tpr_gb_15, tpr_gb_16]
ax.plot(x, tpr_l_gb, color='red', label='Taxa Verdadeiros Positivos (GB)')

#Labels
plt.xlabel("Quantidade de Mineradores", size=16)
plt.ylabel("Taxa de Verdadeiros Positivos (%)", size=16)


plt.legend(loc='upper right', prop={'size': 16})
plt.show()

N = 16
fig, ax = plt.subplots()
width = 0.25

r1 = np.arange(N)
r2 = [x + width for x in r1]



ax.bar(r1, tpr_l_fa, width, color='royalblue', label='Floresta Aleatória')
ax.bar(r2, tpr_l_gb, width, color='red', label='Gradient Booster Tree')

plt.legend()
plt.show()

####################

def get_resources(miner, alg):
    monitor = np.genfromtxt('../process-layer/'+alg+'/'+miner+'-miner/monitor.csv', delimiter=',')
    cpu = [x[0] for x in monitor]
    mem = [x[1] for x in monitor]
    inp = [x[2] for x in monitor]
    oup = [x[3] for x in monitor]

    return cpu,mem,inp,oup
'''
cpu_fa_1, mem_fa_1, inp_fa_1, oup_fa_1 = get_resources('1', 'FA')
cpu_fa_2, mem_fa_2, inp_fa_2, oup_fa_2 = get_resources('2', 'FA')
cpu_fa_3, mem_fa_3, inp_fa_3, oup_fa_3 = get_resources('3', 'FA')
cpu_fa_4, mem_fa_4, inp_fa_4, oup_fa_4 = get_resources('4', 'FA')
cpu_fa_5, mem_fa_5, inp_fa_5, oup_fa_5 = get_resources('5', 'FA')
cpu_fa_6, mem_fa_6, inp_fa_6, oup_fa_6 = get_resources('6', 'FA')
cpu_fa_7, mem_fa_7, inp_fa_7, oup_fa_7 = get_resources('7', 'FA')
cpu_fa_8, mem_fa_8, inp_fa_8, oup_fa_8 = get_resources('8', 'FA')
cpu_fa_9, mem_fa_9, inp_fa_9, oup_fa_9 = get_resources('9', 'FA')
cpu_fa_10, mem_fa_10, inp_fa_10, oup_fa_10 = get_resources('10', 'FA')
cpu_fa_11, mem_fa_11, inp_fa_11, oup_fa_11 = get_resources('11', 'FA')
cpu_fa_12, mem_fa_12, inp_fa_12, oup_fa_12 = get_resources('12', 'FA')
cpu_fa_13, mem_fa_13, inp_fa_13, oup_fa_13 = get_resources('13', 'FA')
cpu_fa_14, mem_fa_14, inp_fa_14, oup_fa_14 = get_resources('14', 'FA')
cpu_fa_15, mem_fa_15, inp_fa_15, oup_fa_15 = get_resources('15', 'FA')
cpu_fa_16, mem_fa_16, inp_fa_16, oup_fa_16 = get_resources('16', 'FA')

cpu_gb_1, mem_gb_1, inp_gb_1, oup_gb_1 = get_resources('1', 'GB')
cpu_gb_2, mem_gb_2, inp_gb_2, oup_gb_2 = get_resources('2', 'GB')
cpu_gb_3, mem_gb_3, inp_gb_3, oup_gb_3 = get_resources('3', 'GB')
cpu_gb_4, mem_gb_4, inp_gb_4, oup_gb_4 = get_resources('4', 'GB')
cpu_gb_5, mem_gb_5, inp_gb_5, oup_gb_5 = get_resources('5', 'GB')
cpu_gb_6, mem_gb_6, inp_gb_6, oup_gb_6 = get_resources('6', 'GB')
cpu_gb_7, mem_gb_7, inp_gb_7, oup_gb_7 = get_resources('7', 'GB')
cpu_gb_8, mem_gb_8, inp_gb_8, oup_gb_8 = get_resources('8', 'GB')
cpu_gb_9, mem_gb_9, inp_gb_9, oup_gb_9 = get_resources('9', 'GB')
cpu_gb_10, mem_gb_10, inp_gb_10, oup_gb_10 = get_resources('10', 'GB')
cpu_gb_11, mem_gb_11, inp_gb_11, oup_gb_11 = get_resources('11', 'GB')
cpu_gb_12, mem_gb_12, inp_gb_12, oup_gb_12 = get_resources('12', 'GB')
cpu_gb_13, mem_gb_13, inp_gb_13, oup_gb_13 = get_resources('13', 'GB')
cpu_gb_14, mem_gb_14, inp_gb_14, oup_gb_14 = get_resources('14', 'GB')
cpu_gb_15, mem_gb_15, inp_gb_15, oup_gb_15 = get_resources('15', 'GB')
cpu_gb_16, mem_gb_16, inp_gb_16, oup_gb_16 = get_resources('16', 'GB')
'''

def taxa_mem(mem):
    mem_tax = []
    for i in range(len(mem)):
        taxa = 100 * mem[i] / 16384
        mem_tax.append(taxa)
    return mem_tax
'''
cpufa = [cpu_fa_1, cpu_fa_2, cpu_fa_3, cpu_fa_4,
     cpu_fa_5, cpu_fa_6, cpu_fa_7, cpu_fa_8,
     cpu_fa_8, cpu_fa_9, cpu_fa_10, cpu_fa_11,
     cpu_fa_12, cpu_fa_13, cpu_fa_14, cpu_fa_15, cpu_fa_16]
cpugb = [cpu_gb_1, cpu_gb_2, cpu_gb_3, cpu_gb_4,
     cpu_gb_5, cpu_gb_6, cpu_gb_7, cpu_gb_8,
     cpu_gb_8, cpu_gb_9, cpu_gb_10, cpu_gb_11,
     cpu_gb_12, cpu_gb_13, cpu_gb_14, cpu_gb_15, cpu_gb_16]

mem_fa_1 = taxa_mem(mem_fa_1)
mem_fa_2 = taxa_mem(mem_fa_2)
mem_fa_3 = taxa_mem(mem_fa_3)
mem_fa_4 = taxa_mem(mem_fa_4)
mem_fa_5 = taxa_mem(mem_fa_5)
mem_fa_6 = taxa_mem(mem_fa_6)
mem_fa_7 = taxa_mem(mem_fa_7)
mem_fa_8 = taxa_mem(mem_fa_8)
mem_fa_9 = taxa_mem(mem_fa_9)
mem_fa_10 = taxa_mem(mem_fa_10)
mem_fa_11 = taxa_mem(mem_fa_11)
mem_fa_12 = taxa_mem(mem_fa_12)
mem_fa_13 = taxa_mem(mem_fa_13)
mem_fa_14 = taxa_mem(mem_fa_14)
mem_fa_15 = taxa_mem(mem_fa_15)
mem_fa_16 = taxa_mem(mem_fa_16)

memfa = [mem_fa_1, mem_fa_2, mem_fa_3, mem_fa_4,
     mem_fa_5, mem_fa_6, mem_fa_7, mem_fa_8,
     mem_fa_8, mem_fa_9, mem_fa_10, mem_fa_11,
     mem_fa_12, mem_fa_13, mem_fa_14, mem_fa_15, mem_fa_16]

mem_gb_1 = taxa_mem(mem_gb_1)
mem_gb_2 = taxa_mem(mem_gb_2)
mem_gb_3 = taxa_mem(mem_gb_3)
mem_gb_4 = taxa_mem(mem_gb_4)
mem_gb_5 = taxa_mem(mem_gb_5)
mem_gb_6 = taxa_mem(mem_gb_6)
mem_gb_7 = taxa_mem(mem_gb_7)
mem_gb_8 = taxa_mem(mem_gb_8)
mem_gb_9 = taxa_mem(mem_gb_9)
mem_gb_10 = taxa_mem(mem_gb_10)
mem_gb_11 = taxa_mem(mem_gb_11)
mem_gb_12 = taxa_mem(mem_gb_12)
mem_gb_13 = taxa_mem(mem_gb_13)
mem_gb_14 = taxa_mem(mem_gb_14)
mem_gb_15 = taxa_mem(mem_gb_15)
mem_gb_16 = taxa_mem(mem_gb_16)

memgb = [mem_gb_1, mem_gb_2, mem_gb_3, mem_gb_4,
     mem_gb_5, mem_gb_6, mem_gb_7, mem_gb_8,
     mem_gb_8, mem_gb_9, mem_gb_10, mem_gb_11,
     mem_gb_12, mem_gb_13, mem_gb_14, mem_gb_15, mem_gb_16]

inpfa = [inp_fa_1, inp_fa_2, inp_fa_3, inp_fa_4,
     inp_fa_5, inp_fa_6, inp_fa_7, inp_fa_8,
     inp_fa_8, inp_fa_9, inp_fa_10, inp_fa_11,
     inp_fa_12, inp_fa_13, inp_fa_14, inp_fa_15, inp_fa_16]
inpgb = [inp_gb_1, inp_gb_2, inp_gb_3, inp_gb_4,
     inp_gb_5, inp_gb_6, inp_gb_7, inp_gb_8,
     inp_gb_8, inp_gb_9, inp_gb_10, inp_gb_11,
     inp_gb_12, inp_gb_13, inp_gb_14, inp_gb_15, inp_gb_16]

oupfa = [oup_fa_1, oup_fa_2, oup_fa_3, oup_fa_4,
     oup_fa_5, oup_fa_6, oup_fa_7, oup_fa_8,
     oup_fa_8, oup_fa_9, oup_fa_10, oup_fa_11,
     oup_fa_12, oup_fa_13, oup_fa_14, oup_fa_15, oup_fa_16]
oupgb = [oup_gb_1, oup_gb_2, oup_gb_3, oup_gb_4,
     oup_gb_5, oup_gb_6, oup_gb_7, oup_gb_8,
     oup_gb_8, oup_gb_9, oup_gb_10, oup_gb_11,
     oup_gb_12, oup_gb_13, oup_gb_14, oup_gb_15, oup_gb_16]


ticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(cpufa, positions=np.array(range(len(cpufa)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(cpugb, positions=np.array(range(len(cpugb)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

plt.plot([], c='#D7191C', label='Floresta Aleatória')
plt.plot([], c='#2C7BB6', label='Gradient Booster Tree')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 8)
plt.ylabel("Carga CPU (%)")
plt.xlabel("Quantidade de Mineradores")
plt.tight_layout()
plt.show()

plt.figure()

bpl = plt.boxplot(memfa, positions=np.array(range(len(memfa)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(memgb, positions=np.array(range(len(memgb)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

plt.plot([], c='#D7191C', label='Floresta Aleatória')
plt.plot([], c='#2C7BB6', label='Gradient Booster Tree')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 8)
plt.ylabel("Memória")
plt.xlabel("Quantidade de Mineradores")
plt.tight_layout()
plt.show()

plt.figure()

bpl = plt.boxplot(inpfa, positions=np.array(range(len(inpfa)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(inpgb, positions=np.array(range(len(inpgb)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

plt.plot([], c='#D7191C', label='Floresta Aleatória')
plt.plot([], c='#2C7BB6', label='Gradient Booster Tree')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 8)
plt.ylabel("Input Packets")
plt.xlabel("Quantidade de Mineradores")
plt.tight_layout()
plt.show()

plt.figure()

bpl = plt.boxplot(oupfa, positions=np.array(range(len(oupfa)))*2.0-0.4, sym='', widths=0.6)
bpr = plt.boxplot(oupgb, positions=np.array(range(len(oupgb)))*2.0+0.4, sym='', widths=0.6)
set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#2C7BB6')

plt.plot([], c='#D7191C', label='Floresta Aleatória')
plt.plot([], c='#2C7BB6', label='Gradient Booster Tree')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)
plt.xlim(-2, len(ticks)*2)
#plt.ylim(0, 8)
plt.ylabel("Output Packets")
plt.xlabel("Quantidade de Mineradores")
plt.tight_layout()
plt.show()
'''