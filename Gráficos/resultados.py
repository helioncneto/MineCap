
from proj import plot_roc_multi
import numpy as np
import matplotlib.pyplot as plt
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


plot_roc_multi(y1=desejada1, y2=desejada2, prob1=proba1, prob2=proba2, alg1='RF', alg2='LR')

N = 3
fig, ax = plt.subplots()
width = 0.25

r1 = np.arange(N)
r2 = [x + width for x in r1]

dt = [precision1, sensibilidade1, especificidade1]
ax.bar(r1, dt, width, color='royalblue', label='Random Forest')
lr = [precision2, sensibilidade2, especificidade2]
ax.bar(r2, lr, width, color='none', edgecolor='lime', hatch = 'xxxx', label='Logistic Regression')

ax.set_xticks([r + width/2 for r in range(len(dt))])
ax.set_xticklabels(('Precisão', 'Sensibilidade', 'Especificidade'), fontsize=16)

plt.legend()
plt.show()



