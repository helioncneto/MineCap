#!/usr/bin/env python
# coding: utf-8

# In[3]:

def get_label(modelo, miner):
    fluxo = []
    arq = open('../process-layer/'+modelo+'/'+miner+'-miner/fluxo_puro.txt', 'r')
    texto = arq.readlines()
    for linha in texto:
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
        with open('../process-layer/'+modelo+'/'+miner+'-miner/lbl_fluxos.txt', 'a') as arq:
            arq.write(str(i))
            arq.write('\n')

#get_label(modelo='LR')
#get_label(modelo='NB')
#get_label(modelo='GB',miner='1')
#get_label(modelo='GB',miner='2')
#get_label(modelo='GB',miner='3')
#get_label(modelo='GB',miner='4')
#get_label(modelo='GB',miner='5')
#get_label(modelo='GB',miner='6')
#get_label(modelo='GB',miner='7')
#get_label(modelo='GB',miner='8')
#get_label(modelo='GB',miner='9')
#get_label(modelo='GB',miner='10')
#get_label(modelo='GB',miner='11')
#get_label(modelo='GB',miner='12')
#get_label(modelo='GB',miner='13')
#get_label(modelo='GB',miner='14')
get_label(modelo='GB',miner='15')
#get_label(modelo='FA',miner='14')

#get_label(modelo='GB')





