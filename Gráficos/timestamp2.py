import numpy as np

arq = open('timestamp/h1_r.txt')
h1_mc = []
testados={}
classificado={}
tempo={}
ips_rest=[]
texto = arq.readlines()
for linha in texto:
    a = linha.split(',')
    if a[0] != '10.0.0.1':
        if a[0] not in testados:
            h1_mc.append(float(a[-1])-float(a[-2]))
            testados[a[0]]=1
        if a[0] not in classificado:
            if a[4]=="1.0":
                classificado[a[0]]=[0,'ok']
                tempo[a[0]] = a[-1][0:-1]
            else:
                classificado[a[0]]=[a[-1],'n']
        else:
            if a[4]=='1.0':
                classificado[a[0]]=[a[-1]-classificado[a[0]][0],'ok']
                tempo[a[0]]=a[-1][0:-1]

    if a[2] != '10.0.0.1':
        if a[2] not in testados:
            h1_mc.append(float(a[-1])-float(a[-2]))
            testados[a[2]]=1
        if a[2] not in classificado:
            if a[4]=="1.0":
                classificado[a[2]]=[0,'ok']
                tempo[a[2]] = a[-1][0:-1]
            else:
                classificado[a[2]]=[a[-1],'n']
        else:
            if a[4]=='1.0':
                classificado[a[2]]=[float(a[-1])-float(classificado[a[2]][0]),'ok']
                tempo[a[2]]=a[-1][0:-1]

#print(tempo)
f = open('timestamp/h1.txt')
for line in f:
    a = line.split()
    ip1 = a[-3].split('\'')[1]
    ip2 = a[-5].split('\'')[1]
    if ip1 != '10.0.0.1':
        if ip1 not in ips_rest:
            ips_rest.append(ip1)
    else:
        if ip2 not in ips_rest:
            ips_rest.append(ip2)
    for i in range(len(ips_rest)):
        if ips_rest[i] in tempo:
            #print (a[-1], tempo[ip])
            tempo[ips_rest[i]] = float(a[-1])-float(tempo[ips_rest[i]])

print(ips_rest)
print (h1_mc)
print (classificado)
print (tempo)
arq.close()


