import Statistics

s = Statistics.Statistics()

listaValores = [5,3,2,1,7,8,5,4,3,6,7,7,7,8]

media = s.getMean(listaValores)
ic = s.getConfidenceInterval(listaValores)

print(media, ic, media-ic, media+ic)