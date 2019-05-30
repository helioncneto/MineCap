def get_label(min):
    fluxo = []
    arq = open('incremental/fluxo_puro'+min+'.txt', 'r')
    texto = arq.readlines()
    for linha in texto:
        a = linha.split(',')
        fluxo.append(a)
    arq.close()


    label = []

    for i in fluxo:
        c = 0

        if int(i[1]) == 14444:
            j = 1.0
            c = 1
        elif int(i[1]) == 2222:
            j = 1.0
            c = 1
        elif int(i[1]) == 3032:
            j = 1.0
            c = 1
        elif int(i[1]) == 8005:
            j = 1.0
            c = 1

        if c != 1:
            if int(i[3]) == 14444:
                j = 1.0
            elif int(i[3]) == 2222:
                j = 1.0
            elif int(i[3]) == 3032:
                j = 1.0
            elif int(i[3]) == 8005:
                j = 1.0
            else:
                j = 0.0
        label.append(j)

    for i in label:
        with open('incremental/lbl_fluxos'+min+'.txt', 'a') as arq:
            arq.write(str(i))
            arq.write('\n')

get_label('_5min')
get_label('_10min')
get_label('_15min')
get_label('_20min')
get_label('_25min')
get_label('_30min')