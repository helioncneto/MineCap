#!/bin/bash

#Declarar os PATHs
zkpath="/home/helio/MineCap/infra/zookeeper-3.4.11"
kfkpath="/home/helio/MineCap/infra/kafka_2.11-0.11.0.0"
capturepath="/home/helio/MineCap/capture-layer"

$zkpath/bin/zkServer.sh status


if [ $? -eq 1 ]
then
$zkpath/bin/zkServer.sh start
fi

$kfkpath/bin/kafka-server-start.sh -daemon $kfkpath/config/server.properties

sleep 4

python2.7 $capturepath/read-network-flowtbag-fromInterface.py &
