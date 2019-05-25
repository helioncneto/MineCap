import numpy as np

def media_timestamp(n_min):
    processados = []
    classificados = []
    rest = []
    for i in range(1,n_min+1):
        arq = open('timestamp/h'+str(i)+'_r.txt')
        h1_mc = []
        testados={}
        classificado={}
        tempo={}
        texto = arq.readlines()
        for linha in texto:
            a = linha.split(',')
            if a[0] != '10.0.0.'+str(i):
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
                        classificados.append(classificado[a[0]][0])
                tempo[a[0]]=a[-1]

            if a[2] != '10.0.0.'+str(i):
                if a[2] not in testados:
                    h1_mc.append(float(a[-1])-float(a[-2]))
                    testados[a[2]]=1
                if a[2] not in classificado:
                    if a[4]=="1.0":
                        classificado[a[2]]=[0,'ok']
                        classificados.append(classificado[a[2]][0])
                    else:
                        classificado[a[2]]=[a[-1],'n']
                else:
                    if a[4]=='1.0':
                        classificado[a[2]]=[float(a[-1])-float(classificado[a[2]][0]),'ok']
                tempo[a[2]]=a[-1][:-1]
        f = open('timestamp/h'+str(i)+'.txt')
        tempo2 = {}
        l_tempo = []
        for line in f:
            a = line.split()
            ip1 = a[-3].split('\'')[1]
            ip2 = a[-5].split('\'')[1]
            if ip1 == '10.0.0.'+str(i):
                ip = ip2
            else:
                ip = ip1
            #print ("ip",ip)
            if ip in tempo and ip not in tempo2:
                #print (a[-1], tempo[ip])
                tempo[ip] = float(a[-1])-float(tempo[ip])
                l_tempo.append(tempo[ip])
                tempo2[ip]=0
        processados.append(h1_mc)
        rest.append(l_tempo)
    print(processados)
    print(classificados)
    print(rest)

    print (h1_mc)
    print (classificado)
    print (tempo)
    arq.close()

media_timestamp(1)


