import os

def add_flow(a, b):
    teste = '''#!/bin/bash
    
    curl -X POST -d '{ \\
        "dpid": 1, \\
        "cookie": 0, \\
        "table_id": 0, \\
        "priority": 100, \\
        "idle_timeout": 300, \\
        "flags": 1, \\
        "match":{ \\
            "ipv4_src": "'''+a+'''", \\
            "ipv4_dst": "'''+b+'''", \\
            "eth_type": 2048 \\
        }, \\
        "actions":[] \\
        }' http://192.168.1.225:8080/stats/flowentry/add'''
    with open('/tmp/add_flow.sh', 'a') as arq:
        arq.write(teste)
        arq.write('\n')

    os.system("chmod +x /tmp/add_flow.sh")
    os.system("sh /tmp/add_flow.sh")
    os.system("rm -f /tmp/add_flow.sh")
