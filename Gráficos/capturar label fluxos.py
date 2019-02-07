#!/usr/bin/env python
# coding: utf-8

# In[3]:


#fluxo = np.genfromtxt('fluxo_puro.txt', delimiter=',')
fluxo = []
arq = open('../process-layer/LR/fluxo_puro.txt', 'r')
texto = arq.readlines()
for linha in texto :
    a = linha.split(',')
    fluxo.append(a)
arq.close()


# In[4]:


label = []

for i in fluxo:
    c = 0
    
    if int(i[1]) == 45700:
        j = 1.0
        c = 1
    elif int(i[1]) == 45550:
        j = 1.0
        c = 1
    elif int(i[1]) == 45560:
        j = 1.0
        c = 1
    elif int(i[1]) == 45791:
        j = 1.0
        c = 1
    
    if c != 1:    
        if int(i[3]) == 45700:
            j = 1.0
        elif int(i[3]) == 45550:
            j = 1.0
        elif int(i[3]) == 45560:
            j = 1.0
        elif int(i[3]) == 45791:
            j = 1.0
        else:
            j = 0.0
    label.append(j)

for i in label:
    with open('lbl_fluxos_LR.txt', 'a') as arq:
        arq.write(str(i))
        arq.write('\n')


# In[ ]:




