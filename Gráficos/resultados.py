
from proj import plot_roc_multi
import numpy as np
desejada1 = np.genfromtxt('lbl_fluxos.txt')
desejada2 = np.genfromtxt('lbl_fluxos_LR.txt')


# In[2]:


prevista1 = np.genfromtxt('../process-layer/DT/outputs.txt')
prevista2 = np.genfromtxt('../process-layer/LR/outputs.txt')

# In[3]:


#Acuracia
from sklearn.metrics import accuracy_score
accuracy_score(desejada1, prevista1)
accuracy_score(desejada2, prevista2)


# In[4]:


from sklearn.metrics import confusion_matrix
tn1, fp1, fn1, tp1 = confusion_matrix(desejada1, prevista1).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(desejada2, prevista2).ravel()


# In[5]:


precision1 = tp1 / (tp1 + fp1)
sensibilidade1 = tp1 / (tp1 + fn1)
especificidade1 = tn1 / (tn1 + fp1)

precision2 = tp2 / (tp2 + fp2)
sensibilidade2 = tp2 / (tp2 + fn2)
especificidade2 = tn2 / (tn2 + fp2)

print("Precisão DT: ", precision1)
print("Sensibilidade DT: ", sensibilidade1)
print("Especificidade DT: ", especificidade1)

print("Precisão LR: ", precision2)
print("Sensibilidade LR: ", sensibilidade2)
print("Especificidade LR: ", especificidade2)


print(confusion_matrix(desejada1, prevista1))
print(confusion_matrix(desejada2, prevista2))


# In[7]:


from sklearn.metrics import precision_score
precision_score(desejada1, prevista1, average='micro')
precision_score(desejada2, prevista2, average='micro')


# In[8]:


proba1 = np.genfromtxt('../process-layer/DT/proba.txt', delimiter=',')
proba2 = np.genfromtxt('../process-layer/LR/proba.txt', delimiter=',')


'''
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


    
a = []
for i in range(len(desejada)):
    if desejada[i] == 0:
        a.append([1, 0])
    else:
        a.append([0, 1])
a = np.array(a)


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(a[:, i], proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(a.ravel(), proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='Curva ROC (AUC = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
#plt.title(titulo)
plt.legend(loc="lower right")
plt.show()
'''

plot_roc_multi(y1=desejada1, y2=desejada2, prob1=proba1, prob2=proba2, alg1='RF', alg2='LR')


