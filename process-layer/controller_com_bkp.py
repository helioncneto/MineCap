import os

#a = "10.0.0.1"
#b = "10.0.0.2"

def add_flow(a, b):
    teste = '''#!/bin/bash
    
    curl -X POST -d '{ \\
        "dpid": 1, \\
        "cookie": 0, \\
        "table_id": 0, \\
        "priority": 100, \\
        "idle_timeout": 30, \\
        "flags": 1, \\
        "match":{ \\
            "nw_src": "'''+a+'''", \\
            "nw_dst": "'''+b+'''", \\
            "dl_type": 2048 \\
        }, \\
        "actions":[] \\
        }' http://localhost:8080/stats/flowentry/add'''
    with open('/tmp/add_flow.sh', 'a') as arq:
        arq.write(teste)
        arq.write('\n')

    os.system("chmod +x /tmp/add_flow.sh")
    os.system("sh /tmp/add_flow.sh")
    os.system("rm -f /tmp/add_flow.sh")
