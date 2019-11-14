def get_label(min, periodo):
    for p in range(1, periodo+1):
        fluxo = []
        arq = open('incremental/fluxo_puro_' + min + '_' + str(p) + '.txt', 'r')
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
            with open('incremental/lbl_fluxos_' + min + '_' + str(p) + '.txt', 'a') as arq:
                arq.write(str(i))
                arq.write('\n')

get_label('0min', 48)
get_label('5min', 48)
get_label('10min', 48)
get_label('15min', 48)
get_label('20min', 48)
get_label('25min', 48)
get_label('30min', 48)