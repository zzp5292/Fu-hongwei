from math import ceil

import networkx as nx
import numpy as np


class GRC:
    def __init__(self, damping, alpha, **kwargs):
        self.name='GRC'
        self.damping = damping
        self.alpha = alpha
        #sgrc代表物理节点的评价,rgrc代表请求的评价分数
        self.sgrc, self.rgrc =  [], []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        # if not self.rgrc:
        #     self.rgrc = self.request_grc(env)
        #
        # vgrc = self.rgrc.pop(0)

        # in contrast to the original paper, we recompute the substrate GRC vector after every placement decision
        # since their setting assumes (1) resource demands irrespective of placements
        # and (2) more than one VNF instance may be served by the same node   
        self.sgrc = self.substrate_grc(env)

        argsort = sorted(range(len(self.sgrc)), key=self.sgrc.__getitem__)
        argsort.reverse()

        action = next(node for node in argsort if node in env.valid_routes)
        return action

    def substrate_grc(self, env):
        num_nodes = len(env.net.nodes())

        # compute (normalized) remaining computing and memory resources
        #计算（标准化）剩余的计算和内存资源
        compute = np.asarray(list(env.computing.values()))
        max_compute = np.asarray(list(c for _, c in env.net.nodes('compute')))
        compute = compute / np.sum(max_compute)

        memory = np.asarray(list(env.memory.values()))
        max_memory = np.asarray(list(m for _, m in env.net.nodes('memory')))
        memory = memory / np.sum(max_memory)

        # compute aggregated resource vector (accounts for multiple resources)
        #计算聚合资源向量（占多个资源）
        resources = self.alpha * compute + (1 - self.alpha) * memory
        vnum = len(env.vtype_bidict.mirror[env.request])
        vtype = env.request.vtypes[vnum]
        config = env.vnfs[vtype]
        incr_list = []
        for node in env.net.nodes():
            # compute resource demand after placing VNF of `node`
            compute, memory = env.compute_resources(node, vtype,place=False)
            max_compute = np.max(np.asarray(list(c for _, c in env.net.nodes('compute'))))
            max_memory = np.max(np.asarray(list(m for _, m in env.net.nodes('memory'))))
            max_data_link=max(data['datarate'] for _, _, data in env.net.edges(data=True))
            cincr=compute/max_compute
            # if (env.computing[node]-cincr)/env.net.nodes[node]['compute']<0.7:
            #     cincr=cincr*5
            mincr=memory/max_memory
            if cincr==0.0:
                config = env.vnfs[vtype]
                prev_cdem, prev_mdem = env.score2(env.rate_list[node][vtype], config, env.request)
                print(env.rate_list[node][vtype])
                print(ceil((env.rate_list[node][vtype][0] + env.rate_list[node][vtype][1] + env.rate_list[node][vtype][2]) / (config['max. req_transf_rate'])))
                env.rate_list[node][vtype][env.request.service] = env.rate_list[node][vtype][
                                                                   env.request.service] + env.request.datarate
                after_cdem, after_mdem = env.score2(env.rate_list[node][vtype], config, env.request)
                print(env.rate_list[node][vtype])
                print(config)
                print(env.request)
                before = [prev_cdem, prev_mdem]
                after = [after_cdem, after_mdem]
                print(before)
                print(after)
                print(env.request)
                raise ValueError("nimabi")
            if node in env.valid_routes:
                if len(env.vtype_bidict.mirror[env.request])==0:
                    # last_node=env.routes_bidict[env.request][-1][1]
                    # # sp=nx.shortest_path_length(env.net,source=last_node,target=node)
                    # lengths, routes = nx.single_source_dijkstra(
                    #     env.net, source=last_node, weight=env.get_weights, cutoff=env.request.resd_lat)
                    # sp=len(routes)-1
                    sp=1
                else:
                    last_node = env.vtype_bidict.mirror[env.request][-1][0]
                    # sp=nx.shortest_path_length(env.net,source=last_node,target=node)
                    lengths, routes = nx.single_source_dijkstra(
                        env.net, source=last_node, weight=env.get_weights, cutoff=env.request.resd_lat)
                    sp = len(routes) - 1
                    if sp==0:
                        sp=1
            else:
                sp=100
                # network_demand=(sp*env.request.datarate)/max_data_link
            incr = (self.alpha * cincr+(1-self.alpha) * mincr)
            incr_list.append(incr)
        for i in range(len(resources)):
            # resources[i]=resources[i]/incr_list[i]
            resources[i] = resources[i]
        # determine datarate transition matrix
        datarate = np.zeros(shape=(num_nodes, num_nodes))
        # for u, v, data in env.net.edges(data=True):
        #     datarate[u, v] = data['datarate']
        #     datarate[v, u] = data['datarate']
        for edge_keys,data in env.datarate.items():
            u,v=edge_keys
            datarate[u, v] = data
            datarate[v, u] = data
        
        # determince grc vector for substrate network
        total_datarate = np.sum(datarate, axis=0)
        datarate = datarate / total_datarate[:, np.newaxis]
        #@矩阵乘法运算符np.linalg.inv矩阵求逆
        substrate_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_nodes) - self.damping * datarate) @ resources
        
        return list(substrate_grc)
    #打分算法，对于请求的VNF，按照VNF长度进行排序
    def request_grc(self, env):
        num_vnfs = len(env.request.vtypes)

        # in our scenario, requested resources depend on the placement, i.e. consider an aggregation of resource demands
        #一维数组，代表了每一种VNF所给当前节点带来的负载变化量
        resources = np.asarray([self._mean_resource_demand(env, env.request, vtype) for vtype in env.request.vtypes])
        resources = resources / np.sum(resources)

        # normalized transition matrix for linear chain of VNFs is the identity matrix
        #VNFs线性链的归一化转移矩阵是单位矩阵
        datarate = np.eye(num_vnfs)
        #np.linalg.inv矩阵求逆,@代表矩阵乘法
        request_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_vnfs) - self.damping * datarate) @ resources
        return list(request_grc)     

    def _mean_resource_demand(self, env, req, vtype):
        config = env.vnfs[vtype]
        demand = []

        for node in env.net.nodes():
            # compute resource demand after placing VNF of `node`
            #supplied_rate代表放置前的总数率,当前节点，指定VNF类型,之前服务的总速率
            supplied_rate = sum([service.datarate for service in env.vtype_bidict[(node, vtype)]])
            after_cdem, after_mdem = env.score(supplied_rate + req.datarate, config)
            prev_cdem, prev_mdem = env.score(supplied_rate, config)
            #计算流增长带来的计算资源的增长量
            cincr = (after_cdem - prev_cdem) / env.net.nodes[node]['compute']
            #计算流增长带来的存储资源的增长量
            mincr = (after_mdem - prev_mdem) / env.net.nodes[node]['memory']

            # compute aggregated increase (accounts for multiple rather than single resource type)
            #计算累计增长（考虑多个而非单一资源类型）
            incr = self.alpha * cincr + (1 - self.alpha) * mincr

            # filter invalid placements (infinite resource demands)
            #筛选无效位置（无限资源需求）对每一个节点都进行了遍历，计算放置一个VNF所可能带来的影响
            if incr >= 0.0:
                demand.append(incr)

        return np.mean(demand)