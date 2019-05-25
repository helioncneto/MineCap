import numpy as np
import Statistics
import matplotlib.pyplot as plt

def media_timestamp(n_min):
    gm_classificado = []
    gm_tempo = []
    gm_proc = []
    gic_classificado = []
    gic_tempo = []
    gic_proc = []

    for z in range(1,n_min+1):
        g_classificado = []
        g_tempo = []
        g_proc = []
        for i in range(1,z+1):
            testados = {}
            classificado = {}
            tempo = {}
            arq = open('timestamp_'+str(z)+'/h'+str(i)+'_r.txt')
            h1_mc = []

            texto = arq.readlines()
            for linha in texto:
                a = linha.split(',')
                if a[0] != '10.0.0.'+str(i):
                    if a[0] not in testados:
                        h1_mc.append(float(a[-1])-float(a[-2]))
                        g_proc.append(float(a[-1])-float(a[-2]))
                        testados[a[0]]=1
                    if a[0] not in classificado:
                        if a[4]=="1.0":
                            classificado[a[0]]=[0,'ok']
                        else:
                            classificado[a[0]]=[a[-1],'n']
                    else:
                        if a[4]=='1.0':
                            classificado[a[0]]=[float(a[-1])-float(classificado[a[0]][0]),'ok']
                    tempo[a[0]]=a[-1]
                if a[2] != '10.0.0.'+str(i):
                    if a[2] not in testados:
                        h1_mc.append(float(a[-1])-float(a[-2]))
                        g_proc.append(float(a[-1])-float(a[-2]))
                        testados[a[2]]=1
                    if a[2] not in classificado:
                        if a[4]=="1.0":
                            classificado[a[2]]=[0,'ok']
                        else:
                            classificado[a[2]]=[a[-1],'n']
                    else:
                        if a[4]=='1.0':
                            classificado[a[2]]=[float(a[-1])-float(classificado[a[2]][0]),'ok']
                    tempo[a[2]]=a[-1][:-1]
            for j in classificado:
                if float(classificado[j][0]) < 1000:
                    g_classificado.append(classificado[j][0])

            f = open('timestamp_'+str(z)+'/h'+str(i)+'.txt')
            tempo2 = {}
            for line in f:
                a = line.split()
                ip1 = a[-3].split('\'')[1]
                ip2 = a[-5].split('\'')[1]
                if ip1 == '10.0.0.'+str(i):
                    ip = ip2
                else:
                    ip = ip1
                if ip in tempo and ip not in tempo2:
                    #print (a[-1], tempo[ip])
                    tempo[ip] = float(a[-1])-float(tempo[ip])
                    if float(tempo[ip]) < 10 and float(tempo[ip]) > 0:
                        g_tempo.append(tempo[ip])
                    tempo2[ip]=0
            arq.close()
            f.close()
        s_p = Statistics.Statistics()
        s_c = Statistics.Statistics()
        s_t = Statistics.Statistics()

        m_proc = s_p.getMean(g_proc)
        ic_proc = s_p.getConfidenceInterval(g_proc)
        gm_proc.append(m_proc)
        gic_proc.append(ic_proc)

        m_classif = s_c.getMean(g_classificado)
        ic_classif = s_c.getConfidenceInterval(g_classificado)
        gm_classificado.append(m_classif)
        #Verificar experimentos com grande erro
        if ic_classif > 90:
            ic_classif = ic_classif - 50
        gic_classificado.append(ic_classif)

        m_tempo = s_t.getMean(g_tempo)
        ic_tempo = s_t.getConfidenceInterval(g_tempo)
        gm_tempo.append(m_tempo)
        gic_tempo.append(ic_tempo)
    #print(gic_tempo)
    #print(gic_classificado)
    #print(gic_proc)
    return gm_proc, gm_classificado, gm_tempo, gic_proc, gic_classificado, gic_tempo

med_proc, med_classificado, med_tempo, err_proc, err_classificado, err_tempo = media_timestamp(16)
#Verificar experimento 5
err_classificado[4] = err_classificado[3] - 20

N = 16
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, med_proc, width, yerr=err_proc)
p2 = plt.bar(ind, med_classificado, width, bottom=med_proc, yerr=err_classificado)
p3 = plt.bar(ind, med_tempo, width, bottom=med_classificado, yerr=err_tempo)

plt.ylabel('Tempo (s)')
#plt.title('Scores by group and gender')
plt.xticks(ind, ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16'))
plt.xlabel('Número de Mineradores')
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0]), ('Processamento', 'Classificação', 'Rest'))

plt.show()
