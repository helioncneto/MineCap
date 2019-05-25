import numpy as np

import matplotlib.pyplot as plt
import Statistics

def tratamento(minerador):
    processamento = []
    classificacao = []
    rest =[]
    for num_min in range(1,minerador+1):
        arq = open('timestamp_'+str(minerador)+'/h'+str(num_min)+'_r.txt')
        h1_mc = []
        testados={}
        classificado={}
        tempo={}
        texto = arq.readlines()
        for linha in texto:
            a = linha.split(',')
            if a[0] != '10.0.0.'+str(num_min):
                if a[0] not in testados:
                    h1_mc.append(float(a[-1])-float(a[-2]))
                    testados[a[0]]=1
                if a[0] not in classificado:
                    if a[4]=="1.0":
                        classificado[a[0]]=[0,'ok']
                    else:
                        classificado[a[0]]=[a[-1],'n']
                else:
                    if a[4]=='1.0':
                        classificado[a[0]]=[a[-1]-classificado[a[0]][0],'ok']
                tempo[a[0]]=a[-1]

            if a[2] != '10.0.0.'+str(num_min):
                if a[2] not in testados:
                    h1_mc.append(float(a[-1])-float(a[-2]))
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
        #print ("teste")
        #print (tempo)
        f = open('timestamp_'+str(minerador)+'/h'+str(num_min)+'.txt')
        tempo2 = {}
        for line in f:
            a = line.split()
            ip1 = a[-3].split('\'')[1]
            ip2 = a[-5].split('\'')[1]
            if ip1 == '10.0.0.'+str(num_min):
                ip = ip2
            else:
                ip = ip1
            #print ("ip",ip)
            if ip in tempo and ip not in tempo2:
                #print (a[-1], tempo[ip])
                tempo[ip] = float(a[-1])-float(tempo[ip])
                tempo2[ip]=0

        f.close()




        arq.close()
        processamento = processamento + h1_mc
        erros_cla = 0
        erros_rest = 0
        #print ("Processamento ",h1_mc)
        for key in classificado:
            if float(classificado[key][0]) < 1000:
                classificacao.append(classificado[key][0])
            else:
                erros_cla +=1
            if float(tempo[key]) < 1000:
                rest.append(tempo[key])
            else:
                erros_rest += 1
        #print ("Tempo de classificacao",classificado)
        #print ("Tempo rest",tempo)
    return processamento, classificacao, rest, erros_cla, erros_rest



#s = Statistics.Statistics()

#listaValores = [5,3,2,1,7,8,5,4,3,6,7,7,7,8]

#media = s.getMean(listaValores)
#ic = s.getConfidenceInterval(listaValores)

#print(media, ic, media-ic, media+ic)

media_p =[]
ic_p = []
media_c =[]
ic_c = []

media_r =[]
ic_r = []

for i in range(14,17):
    p,c,r, ec, er = tratamento(i)
    print ("Fim do tratamento", p)
    print (c)
    print (r)
    print ("Erros c e r", ec, er)

    s_p = Statistics.Statistics()
    media_p.append(s_p.getMean(p))
    ic_p.append(s_p.getConfidenceInterval(p))

    s_c = Statistics.Statistics()
    media_c.append(s_c.getMean(c))
    ic_c.append(s_c.getConfidenceInterval(c))

    s_r = Statistics.Statistics()
    media_r.append(s_r.getMean(r))
    ic_r.append(s_r.getConfidenceInterval(r))


N = 3 #Trocar para 16
#menMeans = (20, 35, 30, 35, 27)
#womenMeans = (25, 32, 34, 20, 25)
#menStd = (2, 3, 4, 1, 2)
#womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

print (media_p)
print (media_c)
print (media_r)

p1 = plt.bar(ind, media_p, width, yerr=ic_p)
p2 = plt.bar(ind, media_c, width, bottom=media_p, yerr=ic_c)
p3 = plt.bar(ind, media_r, width, bottom=media_c, yerr=ic_r)
plt.ylabel('Atraso')
#plt.title('Scores by group and gender')
plt.xticks(ind, ('14', '15','16'))
#plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0],p3[0]), ('Processamento', 'Classificação', 'Rest'))

plt.show()