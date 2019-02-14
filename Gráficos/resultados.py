
from proj import plot_roc_multi
import numpy as np
import matplotlib.pyplot as plt


desejada1 = np.genfromtxt('lbl_fluxos_DT.txt')
desejada2 = np.genfromtxt('lbl_fluxos_LR.txt')
desejada3 = np.genfromtxt('lbl_fluxos_GB.txt')


# In[2]:


prevista1 = np.genfromtxt('../process-layer/DT/outputs.txt')
prevista2 = np.genfromtxt('../process-layer/LR/outputs.txt')
prevista3 = np.genfromtxt('../process-layer/GB/outputs.txt')

# In[3]:


#Acuracia
from sklearn.metrics import accuracy_score
accuracy_score(desejada1, prevista1)
accuracy_score(desejada2, prevista2)
accuracy_score(desejada3, prevista3)


# In[4]:


from sklearn.metrics import confusion_matrix
tn1, fp1, fn1, tp1 = confusion_matrix(desejada1, prevista1).ravel()
tn2, fp2, fn2, tp2 = confusion_matrix(desejada2, prevista2).ravel()
tn3, fp3, fn3, tp3 = confusion_matrix(desejada3, prevista3).ravel()

# In[5]:


precision1 = tp1 / (tp1 + fp1)
sensibilidade1 = tp1 / (tp1 + fn1)
especificidade1 = tn1 / (tn1 + fp1)

precision2 = tp2 / (tp2 + fp2)
sensibilidade2 = tp2 / (tp2 + fn2)
especificidade2 = tn2 / (tn2 + fp2)

precision3 = tp3 / (tp3 + fp3)
sensibilidade3 = tp3 / (tp3 + fn3)
especificidade3 = tn3 / (tn3 + fp3)

print("Precisão DT: ", precision1)
print("Sensibilidade DT: ", sensibilidade1)
print("Especificidade DT: ", especificidade1)

print("Precisão LR: ", precision2)
print("Sensibilidade LR: ", sensibilidade2)
print("Especificidade LR: ", especificidade2)

print("Precisão GB: ", precision3)
print("Sensibilidade GB: ", sensibilidade3)
print("Especificidade GB: ", especificidade3)

print(confusion_matrix(desejada1, prevista1))
print(confusion_matrix(desejada2, prevista2))
print(confusion_matrix(desejada3, prevista3))

# In[7]:


from sklearn.metrics import precision_score
precision_score(desejada1, prevista1, average='micro')
precision_score(desejada2, prevista2, average='micro')


# In[8]:


proba1 = np.genfromtxt('../process-layer/DT/proba.txt', delimiter=',')
proba2 = np.genfromtxt('../process-layer/LR/proba.txt', delimiter=',')
proba3 = np.genfromtxt('../process-layer/GB/proba.txt', delimiter=',')

plot_roc_multi(y1=desejada1, y2=desejada2, y3=desejada3, prob1=proba1, prob2=proba2, prob3=proba3, alg1='RF', alg2='LR', alg3='GB')

N = 3
fig, ax = plt.subplots()
width = 0.25

r1 = np.arange(N)
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

dt = [precision1, sensibilidade1, especificidade1]
ax.bar(r1, dt, width, color='royalblue', label='Random Forest')
lr = [precision2, sensibilidade2, especificidade2]
ax.bar(r2, lr, width, color='none', edgecolor='lime', hatch = 'xxxx', label='Logistic Regression')
gb = [precision3, sensibilidade3, especificidade3]
ax.bar(r3, gb, width, color='none', edgecolor='red', hatch = '////', label='Gradient Booster Tree')

ax.set_xticks([r + width/2 for r in range(len(dt))])
ax.set_xticklabels(('Precisão', 'Sensibilidade', 'Especificidade'), fontsize=16)

plt.legend()
plt.show()




def gen_traf_count(model, host):
    ip_hs = []
    porta_hs = []
    ip_gw = []
    porta_gw = []
    myip = '10.0.0.'+host

    arq = open('../process-layer/'+model+'/dump-h'+host+'.txt', 'r')
    linha = arq.readline()

    while linha:
        pkt = linha.split(' ')
        if pkt[5] != 'igmp' and pkt[5] != 'ICMP':
            if pkt[1] == 'IP':
                pkt2 = pkt[2].split('.')
                pkt3 = pkt[4].split('.')
                ip = pkt2[0]+'.'+pkt2[1]+'.'+pkt2[2]+'.'+pkt2[3]
                porta = pkt3[4][:-1]
                ip_hs.append(ip)
                porta_hs.append(porta)

        linha = arq.readline()
    arq.close()

    arq = open('../process-layer/GB/dump.txt', 'r')
    linha2 = arq.readline()

    while linha2:
        pkt = linha2.split(' ')
        if pkt[5] != 'igmp' and pkt[5] != 'ICMP':
            if pkt[1] == 'IP':
                pkt2 = pkt[2].split('.')
                pkt3 = pkt[4].split('.')
                ip = pkt2[0]+'.'+pkt2[1]+'.'+pkt2[2]+'.'+pkt2[3]
                porta = pkt3[4][:-1]
                ip_gw.append(ip)
                porta_gw.append(porta)

        linha2 = arq.readline()
    arq.close()


    trafego_ger = 0
    trafego_ent = 0

    for i in range(len(porta_hs)):
        if porta_hs[i] == '45700' and ip_hs[i] == myip:
            trafego_ger += 1
        elif porta_hs[i] == '45550' and ip_hs[i] == myip:
            trafego_ger += 1
        elif porta_hs[i] == '45560' and ip_hs[i] == myip:
            trafego_ger += 1
        elif porta_hs[i] == '45791' and ip_hs[i] == myip:
            trafego_ger += 1

    for i in range(len(porta_gw)):
        if porta_gw[i] == '45700' and ip_gw[i] == myip:
            trafego_ent += 1
        elif porta_gw[i] == '45550' and ip_gw[i] == myip:
            trafego_ent += 1
        elif porta_gw[i] == '45560' and ip_gw[i] == myip:
            trafego_ent += 1
        elif porta_gw[i] == '45791' and ip_gw[i] == myip:
            trafego_ent += 1
    return trafego_ger, trafego_ent


trafego_ger5_GB, trafego_ent5_GB = gen_traf_count(model='GB', host='5')
trafego_ger8_GB, trafego_ent8_GB = gen_traf_count(model='GB', host='8')
trafego_ger9_GB, trafego_ent9_GB = gen_traf_count(model='GB', host='9')
trafego_ger13_GB, trafego_ent13_GB = gen_traf_count(model='GB', host='13')


NH = 4
fig, ax = plt.subplots()
width = 0.25

r_1 = np.arange(NH)
r_2 = [x + width for x in r_1]


tg_gb = [trafego_ger5_GB, trafego_ger8_GB, trafego_ger9_GB, trafego_ger13_GB]
ax.bar(r_1, tg_gb, width, color='blue', label='Tráfego Gerado')
te_gb = [trafego_ent5_GB, trafego_ent8_GB, trafego_ent9_GB, trafego_ent13_GB]
ax.bar(r_2, te_gb, width, color='none', edgecolor='lime', hatch = 'xxxx', label='Tráfego Entregue')


ax.set_xticks([r + width for r in range(len(te_gb))])
ax.set_xticklabels(('Host 5', 'Host 8', 'Host 9', 'Host 13'), fontsize=16)

plt.legend()
plt.show()




