import networkx as nx
import numpy as np


class GRC:
    def __init__(self, damping, alpha, **kwargs):
        self.damping = damping
        self.alpha = alpha

        self.sgrc, self.rgrc =  [], []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.rgrc:
            self.rgrc = self.request_grc(env)

        vgrc = self.rgrc.pop(0)

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
        vnum = len(env.vtype_bidict.mirror[env.request])
        vtype = env.request.vtypes[vnum]
        cdemands, mdemands = {}, {}
        if len(env.routes_bidict[env.request])==0:
            for node in env.net.nodes:
                compute, memory = env.compute_resources(node, vtype, place=False)
                cdemands[node] = compute
                mdemands[node] = memory
            valid_nodes = [node for node in env.net.nodes if
                               cdemands[node] < env.computing[node] and env.computing[node] >= 0 and mdemands[node]<env.memory[node] and env.memory[node]>=0]
            env.valid_routes = {node: "the first node" for node in valid_nodes}
        else:
            _, source = env.routes_bidict[env.request][-1]
            lengths, routes = nx.single_source_dijkstra(env.net, source=source, weight=env.get_weights, cutoff=env.request.resd_lat)
            routes = {node: route for node, route in routes.items() if lengths[node] <= env.request.resd_lat}
            for node in routes:
                compute, memory = env.compute_resources(node, vtype,place=False)
                cdemands[node] = compute
                mdemands[node] = memory
        # valid nodes must provision enough compute and memory resources for the deployment
            valid_nodes = [node for node in routes if cdemands[node] <=
                       env.computing[node] and mdemands[node]<env.memory[node] and env.memory[node]>=0]
            env.valid_routes = {node: env.get_edges(route) for node, route in routes.items() if node in valid_nodes}
        num_nodes = len(env.net.nodes())

        # compute (normalized) remaining computing and memory resources
        compute = np.asarray(list(env.computing.values()))
        max_compute = np.asarray(list(c for _, c in env.net.nodes('compute')))
        compute = compute / np.sum(max_compute)

        memory = np.asarray(list(env.memory.values()))
        max_memory = np.asarray(list(m for _, m in env.net.nodes('memory')))
        memory = memory / np.sum(max_memory)
        # compute aggregated resource vector (accounts for multiple resources)
        cdemand = [0 for node in env.net.nodes()]
        mdemand = [0 for node in env.net.nodes()]
        ndemand=[0 for node in env.net.nodes()]
        resources = compute
        for i in env.net.nodes():
            if i in mdemands.keys():
                mdemand[i] = mdemands[i] / 64
            else:
                mdemand[i] = 1000
        for i in env.net.nodes():
            if i in cdemands.keys():
                cdemand[i] = cdemands[i]
            else:
                cdemand[i] = 10
        resource_cost = [0 for i in env.net.nodes()]
        resource_cost_network = [0 for i in env.net.nodes()]
        for i in env.net.nodes():
            if i in valid_nodes:
                if vnum==0:
                    ndemand[i]=0.0
                elif len(env.valid_routes[i])==0:
                    ndemand[i] =3.0
                else:
                    ndemand[i]=float(len(env.valid_routes[i]))
            else:
                ndemand[i]=1000.0
        for i in env.net.nodes:
            resource_cost[i] = cdemand[i]
            resource_cost_network[i]=ndemand[i]
            resource_cost[i]=(resource_cost[i]+0.5*resource_cost_network[i])
        # print(resource_cost)
        resource_final = resources / (resource_cost)

        # compute aggregated resource vector (accounts for multiple resources)
        # resources = self.alpha * compute + (1 - self.alpha) * memory

        # determine datarate transition matrix
        datarate = np.zeros(shape=(num_nodes, num_nodes))
        for (u, v),data in env.datarate.items():
            datarate[u, v] = data
            datarate[v, u] = data

        # determince grc vector for substrate network
        total_datarate = np.sum(datarate, axis=0)
        datarate = datarate / total_datarate[:, np.newaxis]
        substrate_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_nodes) - self.damping * datarate) @ resource_final

        return list(substrate_grc)

    def request_grc(self, env):
        num_vnfs = len(env.request.vtypes)

        # in our scenario, requested resources depend on the placement, i.e. consider an aggregation of resource demands
        resources = np.asarray([self._mean_resource_demand(env, env.request, vtype) for vtype in env.request.vtypes])
        resources = resources / np.sum(resources)

        # normalized transition matrix for linear chain of VNFs is the identity matrix
        datarate = np.eye(num_vnfs)
        request_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_vnfs) - self.damping * datarate) @ resources

        return list(request_grc)

    def _mean_resource_demand(self, env, req, vtype):
        config = env.vnfs[vtype]
        demand = []

        for node in env.net.nodes():
            # compute resource demand after placing VNF of `node`
            supplied_rate = sum([service.datarate for service in env.vtype_bidict[(node, vtype)]])
            after_cdem, after_mdem = env.score(supplied_rate + req.datarate, config)
            prev_cdem, prev_mdem = env.score(supplied_rate, config)

            cincr = (after_cdem - prev_cdem) / env.net.nodes[node]['compute']
            mincr = (after_mdem - prev_mdem) / env.net.nodes[node]['memory']

            # compute aggregated increase (accounts for multiple rather than single resource type)
            incr = self.alpha * cincr + (1 - self.alpha) * mincr

            # filter invalid placements (infinite resource demands)
            if incr >= 0.0:
                demand.append(incr)

        return np.mean(demand)