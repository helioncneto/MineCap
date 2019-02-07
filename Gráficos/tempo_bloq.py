import numpy as np
from datetime import datetime

#resultado = np.genfromtxt('../process-layer/DT/results.txt')
#dump = np.genfromtxt('../process-layer/DT/dump.txt')

arq = open('../process-layer/DT/results.txt', 'r')
linha = arq.readline()

dump = []
ip_orig = []
port_orig = []
ip_dst = []
port_dst = []
classe = []
num_dump = []
tempos = dict()

while linha:
    fluxos = linha.split('-')
    net = fluxos[0].split(',')
    ip_orig.append(net[0][2:-1])
    port_orig.append(net[1][1:])
    ip_dst.append(net[2][2:-1])
    port_dst.append(net[3][1:-2])
    classe.append(fluxos[1][1:-1])
    linha = arq.readline()

arq2 = open('../process-layer/DT/dump.txt', 'r')
linha2 = arq2.readline()

while linha2:
    pkt = linha2.split(' ')
    if pkt[5] != 'igmp' and pkt[5] != 'ICMP':
        if pkt[1] == 'IP':
            tempo = pkt[0].split('.')
            tempo = tempo[0]

            pkt2 = pkt[2].split('.')
            #print(pkt2)

            dump_pto = pkt2[4]
            dump_ipo = pkt2[0]+'.'+pkt2[1]+'.'+pkt2[2]+'.'+pkt2[3]

            pkt3 = pkt[4].split('.')
            dump_ptd = pkt3[4][:-1]
            dump_ipd = pkt3[0]+'.'+pkt3[1]+'.'+pkt3[2]+'.'+pkt3[3]

            dump.append([dump_ipo, dump_pto, dump_ipd, dump_ptd, tempo])



    linha2 = arq2.readline()
arq2.close()
arq.close()

#print(dump[10])
#print(len(dump))

for i in range(len(ip_orig)):
    lista = []
    if classe[i] == '1.0':
        for j in range(len(dump)):
            if ip_orig[i] == dump[j][0] and port_orig[i] == dump[j][1] and ip_dst[i] == dump[j][2] and port_dst[i] == dump[j][3]:
                lista.append(dump[j][4])
            tempos[ip_orig[i] + '_' + ip_dst[i] + '_' + port_dst[i]] = lista



lista_tempos = []
for i in tempos:
    lista_tempos.append([tempos[i][0], tempos[i][-1]])

for i in lista_tempos:
    a = datetime.strptime(i[0], '%H:%M:%S')
    b = datetime.strptime(i[1], '%H:%M:%S')
    result = b-a
    with open('horas.txt', 'a') as arq:
        arq.write(str(result))
        arq.write('\n')


#print(tempos_obj)




