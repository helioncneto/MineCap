# Autor: Helio Cunha
# Descrição: Aplicação que consome o streaming do Apache Kafka e efetua o processamento
#
# Obs: rodar essa aplicação com Pythons 3.5
#

import findspark
#findspark.init('/home/administrador/MineCap/infra/spark-2.3.0-bin-hadoop2.7')
findspark.init('/opt/spark')

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.3.0 pyspark-shell'

import sys
import time
from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from operator import add
import math
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler

n_secs = 1
topic = "test2"

conf = SparkConf().setAppName("RealTimeDetector").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")
ssc = StreamingContext(sc, n_secs)

spSession = SparkSession.builder.master("local").appName("MineCap").config("spark.some.config.option", "some-value").getOrCreate()

kafkaStream = KafkaUtils.createDirectStream(ssc, [topic], {
                        'bootstrap.servers':'localhost:9092', 
                        'group.id':'flow-streaming', 
                        'fetch.message.max.bytes':'15728640',
                        'auto.offset.reset':'largest'})
                        # Group ID is completely arbitrary

lines = kafkaStream.map(lambda x: x[1])
flows = lines.flatMap(lambda line: line.split(" "))#.map(lambda word: (word[1:-1].split(",")))

##### tratamento dos dados

#fluxoRDD = sc.textFile("/home/administrador/MineCap/process-layer/dataset_fluxo_bc.csv")
fluxoRDD = sc.textFile("/home/helio/MineCap/process-layer/dataset_novo.csv")

# Removendo a primeira linha do arquivo (cabeçalho)
firstLine = fluxoRDD.first()
fluxoRDD2 = fluxoRDD.filter(lambda x: x != firstLine)

def transformToNumeric(inputStr) :

    attList = inputStr.split(",")
    #srcip = float(attList[0])
    #srcport = float(attList[1])
    #dstip = float(attList[2])
    #dstport = float(attList[3])
    #proto = 1.0 if attList[4] == "tcp" else 0.0
    total_fpackets = float(attList[5])
    total_fvolume = float(attList[6])
    total_bpackets = float(attList[7])
    total_bvolume = float(attList[8])
    min_fpktl = float(attList[9])
    mean_fpktl = float(attList[10])
    max_fpktl = float(attList[11])
    std_fpktl = float(attList[12])
    min_bpktl = float(attList[13])
    mean_bpktl = float(attList[14])
    max_bpktl = float(attList[15])
    std_bpktl = float(attList[16])
    min_fiat = float(attList[17])
    mean_fiat = float(attList[18])
    max_fiat = float(attList[19])
    std_fiat = float(attList[20])
    min_biat = float(attList[21])
    mean_biat = float(attList[22])
    max_biat = float(attList[23])
    std_biat = float(attList[24])
    duration = float(attList[25])
    min_active = float(attList[26])
    mean_active = float(attList[27])
    max_active = float(attList[28])
    std_active = float(attList[29])
    min_idle = float(attList[30])
    mean_idle = float(attList[31])
    max_idle = float(attList[32])
    std_idle = float(attList[33])
    sflow_fpackets = float(attList[34])
    sflow_fbytes = float(attList[35])
    sflow_bpackets = float(attList[36])
    sflow_bbytes = float(attList[37])
    fpsh_cnt = float(attList[38])
    bpsh_cnt = float(attList[39])
    #furg_cnt = float(attList[40])
    #burg_cnt = float(attList[41])
    total_fhlen = float(attList[42])
    total_bhlen = float(attList[43])
    dscp = float(attList[44])
    classe = float(attList[45])

    linhas = Row(classe = classe, total_fpackets = total_fpackets, total_fvolume = total_fvolume,
                 total_bpackets = total_bpackets, total_bvolume = total_bvolume, min_fpktl = min_fpktl,
                 mean_fpktl = mean_fpktl, max_fpktl = max_fpktl, std_fpktl = std_fpktl, min_bpktl = min_bpktl,
                 mean_bpktl = mean_bpktl, max_bpktl = max_bpktl, std_bpktl = std_bpktl, min_fiat = min_fiat,
                 mean_fiat = mean_fiat, max_fiat = max_fiat, std_fiat = std_fiat, min_biat = min_biat,
                 mean_biat = mean_biat, max_biat = max_biat, std_biat = std_biat, duration = duration,
                 min_active = min_active, mean_active = mean_active, max_active = max_active,
                 std_active = std_active, min_idle = min_idle, mean_idle = mean_idle, max_idle = max_idle,
                 std_idle = std_idle, sflow_fpackets = sflow_fpackets, sflow_fbytes = sflow_fbytes,
                 sflow_bpackets = sflow_bpackets, sflow_bbytes = sflow_bbytes, fpsh_cnt = fpsh_cnt,
                 bpsh_cnt = bpsh_cnt, total_fhlen = total_fhlen,
                 total_bhlen = total_bhlen, dscp = dscp) 

    return linhas

def transformToNumeric2(inputStr) :

    attList = inputStr.split(",")
    srcip = str(attList[0])
    srcport = int(attList[1])
    dstip = str(attList[2])
    dstport = int(attList[3])
    proto = 1.0 if attList[4] == "tcp" else 0.0
    total_fpackets = float(attList[5])
    total_fvolume = float(attList[6])
    total_bpackets = float(attList[7])
    total_bvolume = float(attList[8])
    min_fpktl = float(attList[9])
    mean_fpktl = float(attList[10])
    max_fpktl = float(attList[11])
    std_fpktl = float(attList[12])
    min_bpktl = float(attList[13])
    mean_bpktl = float(attList[14])
    max_bpktl = float(attList[15])
    std_bpktl = float(attList[16])
    min_fiat = float(attList[17])
    mean_fiat = float(attList[18])
    max_fiat = float(attList[19])
    std_fiat = float(attList[20])
    min_biat = float(attList[21])
    mean_biat = float(attList[22])
    max_biat = float(attList[23])
    std_biat = float(attList[24])
    duration = float(attList[25])
    min_active = float(attList[26])
    mean_active = float(attList[27])
    max_active = float(attList[28])
    std_active = float(attList[29])
    min_idle = float(attList[30])
    mean_idle = float(attList[31])
    max_idle = float(attList[32])
    std_idle = float(attList[33])
    sflow_fpackets = float(attList[34])
    sflow_fbytes = float(attList[35])
    sflow_bpackets = float(attList[36])
    sflow_bbytes = float(attList[37])
    fpsh_cnt = float(attList[38])
    bpsh_cnt = float(attList[39])
    #furg_cnt = float(attList[40])
    #burg_cnt = float(attList[41])
    total_fhlen = float(attList[42])
    total_bhlen = float(attList[43])
    dscp = float(attList[44])
    classe = 0.0
# Cria as linhas com os objetos transformados
    linhas = Row(classe = classe, srcip = srcip, srcport = srcport, dstip = dstip, dstport = dstport, proto = proto,
                 total_fpackets = total_fpackets, total_fvolume = total_fvolume,total_bpackets = total_bpackets,
                 total_bvolume = total_bvolume, min_fpktl = min_fpktl, mean_fpktl = mean_fpktl, max_fpktl = max_fpktl,
                 std_fpktl = std_fpktl, min_bpktl = min_bpktl, mean_bpktl = mean_bpktl, max_bpktl = max_bpktl, std_bpktl = std_bpktl,
                 min_fiat = min_fiat, mean_fiat = mean_fiat, max_fiat = max_fiat, std_fiat = std_fiat, min_biat = min_biat,
                 mean_biat = mean_biat, max_biat = max_biat, std_biat = std_biat, duration = duration,
                 min_active = min_active, mean_active = mean_active, max_active = max_active,
                 std_active = std_active, min_idle = min_idle, mean_idle = mean_idle, max_idle = max_idle,
                 std_idle = std_idle, sflow_fpackets = sflow_fpackets, sflow_fbytes = sflow_fbytes,
                 sflow_bpackets = sflow_bpackets, sflow_bbytes = sflow_bbytes, fpsh_cnt = fpsh_cnt,
                 bpsh_cnt = bpsh_cnt, total_fhlen = total_fhlen,
                 total_bhlen = total_bhlen, dscp = dscp)

    return linhas

fluxoRDD3 = fluxoRDD2.map(transformToNumeric)

# Transforma para Dataframe
fluxoDF = spSession.createDataFrame(fluxoRDD3)

def transformaVar(row) :
    obj = (row["classe"], Vectors.dense([row["total_fpackets"], row["total_fvolume"], row["total_bpackets"],
                                        row["total_bvolume"], row["min_fpktl"], row["mean_fpktl"],
                                        row["max_fpktl"], row["std_fpktl"], row["min_bpktl"], row["mean_bpktl"],
                                        row["max_bpktl"], row["std_bpktl"], row["min_fiat"], row["mean_fiat"],
                                        row["max_fiat"], row["std_fiat"], row["min_biat"], row["mean_biat"],
                                        row["max_biat"], row["std_biat"], row["duration"], row["min_active"],
                                        row["mean_active"], row["max_active"], row["std_active"], row["min_idle"],
                                        row["mean_idle"], row["max_idle"], row["std_idle"], row["sflow_fpackets"],
                                        row["sflow_fbytes"], row["sflow_bpackets"], row["sflow_bbytes"],
                                        row["fpsh_cnt"], row["bpsh_cnt"], 
                                        row["total_fhlen"], row["total_bhlen"], row["dscp"]]))

    return obj

def transformaVar2(row) :
    obj = (row["classe"], Vectors.dense([row["total_fpackets"], row["total_fvolume"], row["total_bpackets"],
                                        row["total_bvolume"], row["min_fpktl"], row["mean_fpktl"],
                                        row["max_fpktl"], row["std_fpktl"], row["min_bpktl"], row["mean_bpktl"],
                                        row["max_bpktl"], row["std_bpktl"], row["min_fiat"], row["mean_fiat"],
                                        row["max_fiat"], row["std_fiat"], row["min_biat"], row["mean_biat"],
                                        row["max_biat"], row["std_biat"], row["duration"], row["min_active"],
                                        row["mean_active"], row["max_active"], row["std_active"], row["min_idle"],
                                        row["mean_idle"], row["max_idle"], row["std_idle"], row["sflow_fpackets"],
                                        row["sflow_fbytes"], row["sflow_bpackets"], row["sflow_bbytes"],
              	                        row["fpsh_cnt"], row["bpsh_cnt"], row["furg_cnt"], row["burg_cnt"],
                                        row["total_fhlen"], row["total_bhlen"], row["dscp"]]))
    return obj


fluxoRDD4 = fluxoDF.rdd.map(transformaVar)

fluxoDF = spSession.createDataFrame(fluxoRDD4,["rotulo", "atributos"])

scaler = MinMaxScaler(inputCol="atributos", outputCol="scaledFeatures", min=0.0, max=1.0)
scalerModel = scaler.fit(fluxoDF)
scaledData = scalerModel.transform(fluxoDF)

# Indexação é pré-requisito para Decision Trees
stringIndexer = StringIndexer(inputCol = "rotulo", outputCol = "indexed")
si_model = stringIndexer.fit(scaledData)
obj_final = si_model.transform(scaledData)

# Criando o modelo
rfClassifer = RandomForestClassifier(labelCol = "indexed", featuresCol = "scaledFeatures", probabilityCol = "probability", numTrees=20)
modelo = rfClassifer.fit(obj_final)



def output_rdd(rdd):
	output = []
	fluxo = []
	s_classe = []
	probability = []
	
	if not rdd.isEmpty():
		rdd2 = rdd.map(transformToNumeric2)
		DF = spSession.createDataFrame(rdd2)
		rdd3 = DF.rdd.map(transformaVar)
		DF = spSession.createDataFrame(rdd3, ["rotulo", "atributos"])
		scaler_Model = scaler.fit(DF)
		scaled_Data = scalerModel.transform(DF)
		string_Indexer = StringIndexer(inputCol = "rotulo", outputCol = "indexed")
		si__model = stringIndexer.fit(scaled_Data)
		obj__final = si__model.transform(scaled_Data)
		#print(obj__final.select("scaledFeatures").show(10))	
		predictions = modelo.transform(obj__final)
		#print(predictions.select("prediction", "scaledFeatures").show(10))
		for i in predictions.select("prediction").collect():
			output.append(i["prediction"])
	
		for i in rdd2.collect():
			fluxo.append([i["srcip"], i["srcport"], i["dstip"], i["dstport"]])

		for i in predictions.select("probability").collect():
			probability.append(i["probability"])

		for i in rdd.collect():
			s_classe.append(i)

	#evaluator = MulticlassClassificationEvaluator(predictionCol = "prediction", labelCol = "indexed", metricName = "accuracy")
	#print(evaluator.evaluate(predictions))

	#print(predictions.groupBy("indexed", "prediction").count().show())

		for ln1, ln2, ln3, ln4 in zip(fluxo, output, s_classe, probability):
			with open('results.txt', 'a') as arq:
				arq.write(str(ln1))
				arq.write('	-	')
				arq.write(str(ln2))
				arq.write('\n')
			with open('outputs.txt', 'a') as arq:
				arq.write(str(ln2))
				arq.write('\n')
			with open('fluxo_puro.txt', 'a') as arq:
				arq.write(str(ln3))
				arq.write('\n')
			with open('proba.txt', 'a') as arq:
                                arq.write(str(ln4))
                                arq.write('\n')


flows.foreachRDD(lambda rdd: output_rdd(rdd))

# Inicia o streaming
ssc.start()
ssc.awaitTermination()

