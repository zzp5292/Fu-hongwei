import random

import networkx as nx
import numpy as np

net=nx.read_gpickle("../data/experiments/abilene_2/abilene.gpickle")
for u,d in net.nodes(data=True):
    d['compute']=np.random.uniform(150,200)
    d['memory']=7680
for u,v,w in net.edges(data=True):
    np.random.seed(10)
    w['datarate']=np.random.uniform(500,1000)
    w['propagation']=np.random.uniform(1,2)
nx.write_gpickle(net,"/home/hello/桌面/xiaofu/Futurecoord2/FutureCoord/data/experiments/abilene_2/abilene.gpickle")
net=nx.read_gpickle("/home/hello/桌面/xiaofu/Futurecoord2/FutureCoord/data/experiments/abilene_2/abilene.gpickle")
for u,d in net.nodes(data=True):
    print(d)
for u,v,w in net.edges(data=True):
    print(w)
