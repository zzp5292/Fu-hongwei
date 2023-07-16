import copy
import heapq
import logging
import random
import time
from collections import Counter
from copy import deepcopy
from itertools import chain, islice, combinations_with_replacement
from math import ceil
from typing import List, Dict, Tuple

import gym
import networkx as nx
import numpy as np
import yaml
from gym import spaces
# from more_itertools import peekable
from matplotlib import pyplot as plt
from tabulate import tabulate
from munch import munchify, unmunchify
from coordination.agents.baselines import AllCombinations
from coordination.agents.grc import GRC
from coordination.environment.bidict import BiDict
from coordination.environment.traffic import Traffic, Request, TrafficStub

#虚拟网络映射版本0-1判断版本(danduilie)
class ServiceCoordination3(gym.Env):
    # substrate network traffic request vnfs services
    def __init__(self, net_path: str, process: Traffic, vnfs: List, services: List):
        self.net_path = net_path
        self.net = nx.read_gpickle(self.net_path)
        self.NUM_NODES = self.net.number_of_nodes()
        self.MAX_DEGREE = max([deg for _, deg in self.net.degree()])
        self.REJECT_ACTION = self.NUM_NODES + 1
        # GLOBAL attribute
        self.MAX_COMPUTE = self.net.graph['MAX_COMPUTE']
        self.MAX_LINKRATE = self.net.graph['MAX_LINKRATE']  # in MB/s
        self.MAX_MEMORY = self.net.graph['MAX_MEMORY']  # in MB
        self.HOPS_DIAMETER = self.net.graph['HOPS_DIAMETER']  # in ms
        self.PROPAGATION_DIAMETER = self.net.graph['PROPAGATION_DIAMETER']  # in ms
        # discrete event generator
        self.process: Traffic = process
        self.PROCESS_STUB = TrafficStub(self.process.sample())
        # 8 queue in a list
        self.all_waiting_queues = []
        self.vnfs: List[dict] = vnfs
        self.services: List[List[int]] = services
        self.NUM_SERVICES = len(self.services)
        self.MAX_SERVICE_LEN = max([len(service) for service in self.services])
        # current source
        self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
        # 强行截断
        self.MAX_QUEUE_LENGTH = 100
        # admission_control
        self.admission = {'deployed': False, 'finalized': False}
        # 我选队列来部署，不决定是否拒绝
        # use 8 discrete action
        self.action_space = spaces.Discrete(2)
        self.ACTION_DIM=2
        # state_space:substrate_network(cpu,mem,network)*Queue(等待队列程度)*成功率(当前成功率)
        self.NETWORK_SIZE = 3
        # 新到请求，被拒绝请求
        self.QUEUE_SIZE = 4
        self.SUCCUESS_SIZE = 4
        self.num_requests = 0
        # 4类业务
        self.SUCCUESS_SIZE = 4
        self.update_interval = 0
        self.invalid_num=0
        # TODO:加入网络温度
        self.OBS_SIZE = 65
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float16)
        # setup basic debug logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(logging.StreamHandler())
        # setup layout for rendering functionality; setup step info
        self.pos = None
        self.info = None
        # it is used to save the all the new-arrival requests
        self.reward = 0
        # current time window
        self.start_time = 0
        self.TIME_WINDOW_LENGTH = 45
        self.end_time = self.start_time + self.TIME_WINDOW_LENGTH
        self.TIME_HORIZON = 45
        self.valid_actions = {}
        self.done1=False
        self.all_requests=0
        self.rate_init()
        self.reward_list=[]
        # 重置环境
        self.reset()
    def compute_accepts(self):
        accepts=[self.info[i].accepts for i in range(4)]
        return accepts

    def compute_reward_no_cost(self):
        return 0.6*(self.info[0].accepts)+0.3*(self.info[1].accepts)+1.0*(self.info[2].accepts)+1.5*(self.info[3].accepts)

    def compute_jain(self):
        accept_rate=self.calculate_accept_rate()
        prior_value=[2,1,3,4]
        jain_every=[]
        if accept_rate[0]==0 and accept_rate[1]==0 and accept_rate[2]==0 and accept_rate[3]==0:
            return 0
        for i in range(4):
            jain_every.append(accept_rate[i]/prior_value[i])
        jain_fenzi=0
        jain_fenmu=0
        for jain in jain_every:
            jain_fenzi+=jain
        jain_fenzi=jain_fenzi**2
        for jain in jain_every:
            jain_fenmu+=jain**2
        jain_fenmu=jain_fenmu*4
        jain_final=jain_fenzi/jain_fenmu
        if jain_final>1.0:
            raise ValueError("Jain error")
        return jain_final

    def time_window(self, start_time, end_time):
        for request in self.PROCESS_STUB:
            # THE NEW ATTIVAL REQUESTS WILL BE INITIALIZED IN ARRRIVAL TIME.HOWEVER WHEN REJECTED,SOME PROPERTIES WILL BE REMOVED
            # WE WILL USE initial_one_REJECT_request FUMCTION TO REINITIAL IT
            if request.arrival >= start_time and request.arrival < end_time:
                #     if request.service == 0:
                #         self.all_waiting_queues[0].append(request)
                #     if request.service == 1:
                #         self.all_waiting_queues[1].append(request)
                #     if request.service == 2:
                #         self.all_waiting_queues[2].append(request)
                #     if request.service == 3:
                #         self.all_waiting_queues[3].append(request)
                self.all_waiting_queues.append(request)
                self.initial_one_request(request)
    def reset(self):
        # 离散事件发生器
        # self.trace = peekable(iter(self.process))
        self.done = False
        self.time = 0
        self.invalid_num = 0
        self.update_interval = 0
        self.request_list = []
        self.reject_list = []
        self.start_time = 0
        self.reward_list = []
        self.end_time = self.start_time + self.TIME_WINDOW_LENGTH
        self.all_waiting_queues = []
        # record the datarate in every instance of every service_type in everynode!
        # 这个就是给monitor模块用的，step用于返回信息。最后的info
        KEYS = ['accepts', 'requests', 'skipped_on_arrival', 'no_egress_route', 'no_extension', 'num_rejects',
                'num_invalid', 'cum_service_length', 'cum_route_hops', 'cum_compute',
                'cum_memory', 'cum_datarate', 'cum_max_latency', 'cum_resd_latency']
        self.info = [munchify(dict.fromkeys(KEYS, 0.0))
                     for _ in range(len(self.services))]

        # 重置底层网络
        self.computing = {node: data['compute']
                          for node, data in self.net.nodes(data=True)}
        self.memory = {node: data['memory']
                       for node, data in self.net.nodes(data=True)}
        self.datarate = {frozenset(
            {src, trg}): data['datarate'] for src, trg, data in self.net.edges(data=True)}

        # save propagation delays for each edge & processing delay per node
        self.propagation = {frozenset(
            {src, trg}): data['propagation'] for src, trg, data in self.net.edges(data=True)}

        # 请求的到达时间和请求的对象实例
        self.deployed, self.valid_routes = [], {}
        # 记录当前部署的节点情况
        self.vtype_bidict = BiDict(None, val_btype=list)
        self.vtype_bidict = BiDict(self.vtype_bidict, val_btype=list)
        # 记录请求和路径节点的映射
        self.routes_bidict = BiDict(None, val_btype=list, key_map=lambda key: frozenset(key))
        self.routes_bidict = BiDict(self.routes_bidict, val_btype=list)
        # capture the first time_window requests
        self.time_window(self.start_time, self.end_time)
        # self.update_interval_time()
        # 寻找有效的动作
        # self.valid_actions = self.valid_actions_method()
        result = self.compute_state_admission()
        # self.compute_state_ad()
        self.initial_cpu = result[4]
        self.initial_memory = result[5]
        self.initial_network = result[6]
        self.rate_init()
        self.all_requests=0
        self.done1=False
        # # print("初始网络状态")
        # self.compute_network_util()
        # 计算熵
        # self.init_entroy()
        return self.compute_state_admission()

        # this function should return the next_placement_state and the placement result

    def place_single_vnf(self, action):
        rejected = (action == self.REJECT_ACTION)
        # reset tracked information for prior request when `action` deploys the next service's initial component
        if not self.request in self.vtype_bidict.mirror:
            self.occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': 0.0}
            self.admission = {'deployed': False, 'finalized': False}

        # check whether action is valid; terminate episode otherwise
        if not (action in self.valid_routes or rejected):
            # do not allow invalid actions for planning agents

            # terminate the episode for model-free agents
            # only means this request has finished
            # an invalid action we should reject it immediately the result should be false
            self.info[self.request.service].num_invalid += 1
            self.logger.debug('Invalid action.')
            reward_step = 0
            result = False
            self.done1=True
            return [], reward_step, result

        # action voluntarily releases partial service embedding; progress to next service

        # the given action extends the partial embedding by another VNF placement
        final_placement = self.place_vnf(action)

        # case: placement of final VNF; check whether valid route to egress node exists
        if final_placement:
            try:
                # compute valid shortest path (latency) routing towards egress
                # NOTE: throws error if not reachable or shortest path has (cumulated) weight > cutoff
                # _, route = nx.single_source_dijkstra(self.net, source=action, target=self.request.egress,
                #                                      weight=self.get_weights, cutoff=self.request.resd_lat)
                # # case: a valid route to the service's egress node exists; service deployment successful
                # # update network state, i.e. steer traffic towards egress
                # route = ServiceCoordination7.get_edges(route)
                # self.steer_traffic(route)

                # register successful service embedding for deletion after duration passed
                exit_time = self.end_time + self.request.duration
                self.deployed.append([exit_time,self.request])
                self.deployed.sort(key=lambda s:s[0])
                # update meta-information for deployed service before progressing in time
                self.update_info()
                self.done1=True
                # print("接受,成功了")
                # # print("_________________________")
                # # print(self.request)
                # # print("接受请求"+str(self.request.service))
                # # print("服务速率"+str(self.request.datarate))
                # # print("剩余时长"+str(self.request.resd_lat))
                # # print("到达时间"+str(self.request.arrival))
                # # print("等待时间"+str(self.request.waiting_time))
                # # print("当前网络状态"+str(self.compute_state_admission()))
                # # print("_______________________________")
                self.logger.debug('Service deployed successfully.')
                self.admission = {'deployed': True, 'finalized': True}

                # progress in time after successful deployment; (implicit action cache update)
                reward_step = self.compute_reward(True, True, self.request,"accept_rate")
                # self.done = self.progress_time()
                result = True
                return [], reward_step, result

            except nx.NetworkXNoPath:
                # case: no valid route to the service's egress node exists
                self.done1=True
                self.info[self.request.service].no_egress_route += 1
                self.logger.debug('No valid route to egress remains.')
                self.admission = {'deployed': False, 'finalized': True}

                # release service; progress in time to next service; (implicit action cache update)
                self.release2(self.request)
                reward = self.compute_reward(True, False, self.request,"accept_rate")
                # self.done = self.progress_time()
                result = False
                # print("拒绝,中途没有合法路由")
                # # print("拒绝请求：原因：与最后一跳没有合法路由")
                # # print("_________________________")
                # # print("离开时间" + str(self.request.departure_time))
                # # print("当前时间" + str(self.end_time))
                # # print("拒绝请求" + str(self.request.service))
                # # print("服务速率" + str(self.request.datarate))
                # # print("剩余时长" + str(self.request.resd_lat))
                # # print("到达时间" + str(self.request.arrival))
                # # print("当前网络状态" + str(self.compute_state_admission()))
                # # print("_______________________________")
                return [], reward, result

        # case: partial embedding not finalized; at least one VNF placement remains to be determined
        reward_step = self.compute_reward(False, False, self.request,"accept_rate")
        self.update_actions()
        # case: partial embedding cannot be extended farther after placement action; proceed to next service
        if not self.valid_routes:
            # print("拒绝,找不到合理的下一跳节点")
            # # print("拒绝请求：原因：没有可用节点放置VNF或者没有相应的路径")
            # # print("_________________________")
            # # print("离开时间" + str(self.request.departure_time))
            # # print("当前时间" + str(self.end_time))
            # # print("拒绝请求" + str(self.request.service))
            # # print("服务速率" + str(self.request.datarate))
            # # print("剩余时长" + str(self.request.resd_lat))
            # # print("到达时间" + str(self.request.arrival))
            # # print("当前网络状态" + str(self.compute_state_admission()))
            # # print("_______________________________")
            self.done1=True
            self.info[self.request.service].no_extension += 1
            self.logger.debug('Cannot extend partial embedding farther.')
            self.admission = {'deployed': False, 'finalized': True}
            # progress in time after the previous service was released; (implicit action cache update)
            self.release2(self.request)
            # self.done = self.progress_time()
            reward_step = self.compute_reward(False, False, self.request,"accept_rate")
            result = False
            return [], reward_step, result
        # case: valid actions remain after extension of partial embedding
        result = True
        self.logger.debug('Proceed with extension of partial embedding.')
        return self.compute_state(), reward_step, result

        # 更换请求

    def replace_process(self, process):
        '''Replace traffic process used to generate request traces.'''
        self.process = process
        self.PROCESS_STUB = TrafficStub(self.process.sample())
        self.reset()

    def compute_reward(self, finalized: bool, deployed: bool, req: Request,calculate_method) -> float:
        '''Reward agents upon the acceptance of requested services.'''
        # todo: calculate the reward by the request,the basic is accrding to the text.it should be (profit),(utilazation),(fairness)
        if deployed:
            # return 1.0
            # if req.waiting_time == 0:
            #     req.waiting_time = 1
            placed = self.vtype_bidict.mirror[self.request]
            crelease = 0.0
            mrelease = 0.0
            routes_length = 0
            for node, vtype in placed:
                config = self.vnfs[vtype]
                supplied_rate = sum([service.datarate for service in self.vtype_bidict[(node, vtype)]])
                prev_cdem, prev_mdem, after_cdem, after_mdem = self.compute_rate_release_2(node, vtype, config, req,
                                                                                           place=False)
                crelease += prev_cdem - after_cdem
                mrelease += prev_mdem - after_mdem
            routes_length += len(self.routes_bidict[self.request])
            network_cost = self.request.datarate * routes_length
            if self.request.service == 3:
                reward = 1.5
            elif self.request.service == 1:
                reward = 0.3
            elif self.request.service == 2:
                reward = 1.0
            else:
                reward = 0.6
            self.reward_list.append(reward)
            if calculate_method == "accept_rate":
                return 1.0
            elif calculate_method == "reward_no_cost":
                if self.request.service == 3:
                    return 4.0
                elif self.request.service == 1:
                    return 1.0
                elif self.service == 2:
                    return 3.0
                else:
                    return 2.0
            else:
                return reward
        return 0.0

    def release(self, req: Request) -> None:
        '''Release network resources bound to the request.'''
        # case: to-be-deployed request is rejected upon its arrival
        if req not in self.vtype_bidict.mirror:
            return

        # release compute & memory resources at nodes with VNF instances that serve the request
        # Counter({(46, 0): 1, (9, 2): 1, (12, 5): 1, (42, 4): 1})
        serving_vnfs = Counter(self.vtype_bidict.mirror[req])
        for (node, vtype), count in serving_vnfs.items():
            config = self.vnfs[vtype]

            # NOTE: account for sharing of VNFs when computing the updated rates by the counter
            supplied_rate = sum(
                [req.datarate for req in self.vtype_bidict[(node, vtype)]])
            updated_rate = supplied_rate - count * req.datarate

            # score resources by difference in scores after datarate has been released
            prev_cdem, prev_mdem = self.score(supplied_rate, config)
            after_cdem, after_mdem = self.score(updated_rate, config)

            # return the computing result
            self.computing[node] += prev_cdem - after_cdem
            # return the memory result
            self.memory[node] += prev_mdem - after_mdem

        # remove to-be-released request from mapping
        del self.vtype_bidict.mirror[req]
        # release datarate along routing path and update datastructure
        route = self.routes_bidict.pop(req, [])
        for src, trg in route[1:]:
            self.datarate[frozenset({src, trg})] += req.datarate

    def release2(self, req: Request) -> None:
        '''Release network resources bound to the request.'''
        # case: to-be-deployed request is rejected upon its arrival
        if req not in self.vtype_bidict.mirror:
            return

        # release compute & memory resources at nodes with VNF instances that serve the request
        # Counter({(46, 0): 1, (9, 2): 1, (12, 5): 1, (42, 4): 1})
        serving_vnfs = Counter(self.vtype_bidict.mirror[req])
        for (node, vtype), count in serving_vnfs.items():
            config = self.vnfs[vtype]

            # # NOTE: account for sharing of VNFs when computing the updated rates by the counter
            # supplied_rate = sum(
            #     [req.datarate for req in self.vtype_bidict[(node, vtype)]])
            # updated_rate = supplied_rate - count * req.datarate
            #
            # # score resources by difference in scores after datarate has been released
            # prev_cdem, prev_mdem = self.score(supplied_rate, config)
            # after_cdem, after_mdem = self.score(updated_rate, config)
            prev_cdem,prev_mdem,after_cdem,after_mdem=self.compute_rate_release_2(node,vtype,config,req,place=True)

            # return the computing result
            self.computing[node] += prev_cdem - after_cdem
            # return the memory result
            self.memory[node] += prev_mdem - after_mdem

        # remove to-be-released request from mapping
        del self.vtype_bidict.mirror[req]
        # release datarate along routing path and update datastructure
        route = self.routes_bidict.pop(req, [])
        for src, trg in route[1:]:
            self.datarate[frozenset({src, trg})] += req.datarate
            req.resd_lat += self.propagation[frozenset({src, trg})]

        # 当某一个请求结束时，更新统计数据

    def update_info(self) -> None:
        service = self.request.service
        self.info[service].accepts += 1
        self.info[service].cum_service_length += len(self.request.vtypes)
        self.info[service].cum_route_hops += len(self.routes_bidict[self.request])
        self.info[service].cum_datarate += self.request.datarate
        self.info[service].cum_max_latency += self.request.max_latency
        self.info[service].cum_resd_latency += self.request.resd_lat

    def render(self, mode: str = 'None', close: bool = False) -> str:
        if mode == 'None' or self.done:
            return

        elif mode == 'textual':
            rtable = [['Compute'] + [str(round(c, 2))
                                     for _, c in self.computing.items()]]
            rtable += [['Memory'] + [str(round(m, 2))
                                     for _, m in self.memory.items()]]

            tnodes = [{t for t in range(len(self.vnfs)) if (
                n, t) in self.vtype_bidict} for n in self.net.nodes()]
            rtable += [['Type'] + tnodes]

            services = [len([num for num, service in enumerate(self.vtype_bidict.mirror) for t in range(
                len(self.vnfs)) if (n, t) in self.vtype_bidict.mirror[service]]) for n in self.net.nodes()]
            rtable += [['Service'] + services]
            cheaders = ['Property'] + [str(node)
                                       for node, _ in self.computing.items()]
            ctable = tabulate(rtable, headers=cheaders, tablefmt='github')
            cutil = 1 - np.mean([self.computing[n] / self.net.nodes[n]
            ['compute'] for n in self.net.nodes])
            mutil = 1 - \
                    np.mean([self.memory[n] / self.net.nodes[n]['memory']
                             for n in self.net.nodes])

            max_cap = [self.net.edges[e]['datarate'] for e in self.net.edges]
            cap = [self.datarate[frozenset({*e})] for e in self.net.edges]
            dutil = np.asarray(cap) / np.asarray(max_cap)
            dutil = 1 - np.mean(dutil)
            graph_stats = f'Util (C): {cutil}; Util (M): {mutil}; Util (D): {dutil}'

            vnum = len(self.vtype_bidict.mirror[self.request])
            vtype = self.request.vtypes[vnum]
            str_repr = '\n'.join(
                (ctable, f'Time: {self.time}', f'Request: {str(self.request)}->{vtype}',
                 f'Available Routes: {self.valid_routes}', f'Graph Stats: {graph_stats}' '\n\n'))
            return str_repr

        if self.pos is None:
            self.pos = nx.spring_layout(self.net, iterations=400)

        # Render the environment to the screen
        edges = {edge: frozenset({*edge}) for edge in self.net.edges}
        datarate = {edge: self.datarate[edges[edge]]
                    for edge in self.net.edges}
        propagation = {
            edge: self.propagation[edges[edge]] for edge in self.net.edges}

        def link_rate(edge):
            return round(datarate[edge], 2)

        def delay(edge):
            return round(propagation[edge], 2)

        _, service_pos = self.routes_bidict[self.request][-1]
        valid_placements = self.valid_routes.keys()

        def color(node):
            if node in valid_placements and node == service_pos:
                return 'cornflowerblue'
            elif node in valid_placements:
                return 'seagreen'
            elif node == service_pos:
                return 'red'
            return 'slategrey'

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        color_map = [color(node) for node in self.net.nodes]
        edge_labels = {edge: (link_rate(edge), delay(edge))
                       for edge in self.net.edges}
        node_labels = {node: (node, round(self.computing[node], 2), round(
            self.memory[node], 2)) for node in self.net.nodes()}
        nx.draw(self.net, self.pos, labels=node_labels, node_color=color_map,
                with_labels=True, node_size=400, ax=ax)
        nx.draw_networkx_edge_labels(self.net, self.pos, edge_labels=edge_labels, ax=ax)

        # 如果剩余带宽可以满足，那么返回传播时延

    def get_weights(self, u: int, v: int, d: Dict) -> float:
        '''Link (propagation) delay invoked when steering traffic across the edge.'''

        # check if it does not provision sufficient datarate resources
        if self.datarate[frozenset({u, v})] < self.request.datarate:
            return None
        # compute propagation & queuing delay based on link utilization & requested datarate
        delay = self.propagation[frozenset({u, v})]
        if self.datarate[frozenset({u, v})]/self.net.edges[frozenset({u, v})]['datarate']<=0.5:
            delay+=self.datarate[frozenset({u, v})]/self.net.edges[frozenset({u, v})]['datarate']*self.propagation[frozenset({u, v})]
        return delay

    def get_weights2(self, u: int, v: int, d: Dict) -> float:
        '''Link (propagation) delay invoked when steering traffic across the edge.'''

        # check if it does not provision sufficient datarate resources
        if self.datarate[frozenset({u, v})] < self.request.datarate:
            return None

        # compute propagation & queuing delay based on link utilization & requested datarate
        datarate = self.datarate[frozenset({u, v})]
        return datarate

    def compute_resources(self, node: int, vtype: int,place) -> Tuple[int]:
        '''Calculate increased resource requirements when placing a VNF of type `vtype` on `node`.'''
        # calculate the datarate served by VNF `vtype` before scheduling the current flow to it
        config = self.vnfs[vtype]
        # the datarate doesn't distinguish the flow type
        prev_cdem, prev_mdem, after_cdem, after_mdem = self.compute_rate_demand_2(node, vtype, config, self.request,place)
        before=[prev_cdem,prev_mdem]
        after=[after_cdem,after_mdem]
        compute, memory = np.subtract(after, before)
        return compute, memory

    # we should check if it is a valid placement_action,so we should compute_node_state,we use it as a function of placement
    def compute_node_state(self, node) -> np.ndarray:
        '''Define node level statistics for state representation.'''
        if not node in self.valid_routes:
            nstate = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            return nstate
        if self.valid_routes[node]=="the first node":
            nstate = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
            return nstate
        # (1) node is a valid placement & valid route to node exists?
        valid = float(node in self.valid_routes)

        # (2) residual latency of request after placement on node
        route = self.valid_routes[node]
        latency = sum([self.propagation[frozenset({u, v})] for u, v in route])
        # if self.request.resd_lat == 0.0: self.request.resd_lat = 1.0
        latency = np.nanmin([latency / self.request.resd_lat, 1.0])

        # (3) quantify datarate demands for placement with #hops to node
        hops = len(route) / self.HOPS_DIAMETER

        # (4) increase of allocated compute/memory after potential placement on node
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        config = self.vnfs[vtype]
        # supplied_rate = sum(
        #     [service.datarate for service in self.vtype_bidict[(node, vtype)]])
        # supplied_rate = sum(
        #     [service.datarate for service in self.vtype_bidict[(node, vtype)]])
        #
        # after_cdem, after_mdem = self.score(
        #     supplied_rate + self.request.datarate, config)
        # prev_cdem, prev_mdem = self.score(supplied_rate, config)
        prev_cdem,prev_mdem,after_cdem,after_mdem=self.compute_rate_demand_2(node,vtype,config,self.request,place=False)
        # calculate the cdemand
        cdemand = np.clip((after_cdem - prev_cdem) /
                          self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        # calculate the mdemand
        mdemand = np.clip((after_mdem - prev_mdem) /
                          self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        # (5) residual compute/memory after placement on node
        resd_comp = np.clip(
            (self.computing[node] - cdemand) / self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
        resd_mem = np.clip(
            (self.memory[node] - mdemand) / self.MAX_MEMORY, a_min=0.0, a_max=1.0)

        return [valid, latency, hops, cdemand, mdemand, resd_comp, resd_mem]

    # this is the satate used by the placement,however it is not used in the step function
    def compute_state(self) -> np.ndarray:
        '''Compute state representation of environment for RL agent.'''
        node_stats = [self.compute_node_state(node) for node in self.net.nodes]
        node_stats = list(chain(*node_stats))

        # encode request level statistics:
        # (1) encode resources bound to partial embedding; i.e. resources that will be released by REJECT_ACTION
        vnum = len(self.vtype_bidict.mirror[self.request])
        placed = self.vtype_bidict.mirror[self.request]
        crelease, mrelease = 0.0, 0.0
        for node, vtype in placed:
            config = self.vnfs[vtype]
            prev_cdem,prev_mdem,after_cdem,after_mdem=self.compute_rate_release_2(node,vtype,config,self.request,place=False)
            crelease += prev_cdem - after_cdem
            mrelease += prev_mdem - after_mdem
        # normalize,crelease,mrelease
        crelease /= self.MAX_SERVICE_LEN * self.MAX_COMPUTE
        mrelease /= self.MAX_SERVICE_LEN * self.MAX_MEMORY

        # (2) one-hot encoding of requested service type
        stype = [1.0 if service ==
                        self.request.service else 0.0 for service in self.services]

        # (3) count encoding of to-be-deployed VNFs for requested service
        counter = Counter(self.request.vtypes[vnum:])
        vnf_counts = [counter[vnf] /
                      self.MAX_SERVICE_LEN for vnf in range(len(self.vnfs))]

        # (4) noramlized datarate demanded by service request
        datarate = self.request.datarate / self.MAX_LINKRATE

        # (5) residual end-to-end latency
        resd_lat = self.request.resd_lat / self.PROPAGATION_DIAMETER

        # (6) one-hot encoding of request's egress node
        # vnf_counts represent the remaining vnf types
        service_stats = [crelease, mrelease, *stype, *vnf_counts, datarate, resd_lat]
        # (1) number of deployed instances for each type of VNF
        num_deployed = [sum([float((node, vtype) in self.vtype_bidict)
                             for node in self.net.nodes]) for vtype in range(len(self.vnfs))]
        num_deployed = [count / len(self.net.nodes) for count in num_deployed]
        mean_cutil = np.mean(
            [self.computing[node] / self.MAX_COMPUTE for node in self.net.nodes])
        mean_mutil = np.mean(
            [self.memory[node] / self.MAX_MEMORY for node in self.net.nodes])
        mean_lutil = np.mean([self.datarate[frozenset(
            {src, trg})] / self.MAX_LINKRATE for src, trg in self.net.edges])

        graph_stats = [*num_deployed, mean_cutil, mean_mutil, mean_lutil]

        return np.asarray(list(chain(node_stats, service_stats, graph_stats)))

    @staticmethod
    def get_edges(nodes: List) -> List:
        return list(zip(islice(nodes, 0, None), islice(nodes, 1, None)))

    @staticmethod
    def score(rate, config):
        if rate <= 0.0:
            return (0.0, 0.0)

            # VNFs cannot serve more than their max. transfer rate (in MB/s)
        elif rate > config['max. req_transf_rate']:
            return (np.inf, np.inf)
            # average rate only one!
        rate = rate / config['scale']
            # score VNF resources by polynomial fit,非线性关系
        compute = config['coff'] + config['ccoef_1'] * rate + config['ccoef_2'] * \
                      (rate ** 2) + config['ccoef_3'] * \
                      (rate ** 3) + config['ccoef_4'] * (rate ** 4)
        memory = config['moff'] + config['mcoef_1'] * rate + config['mcoef_2'] * \
                     (rate ** 2) + config['mcoef_3'] * \
                     (rate ** 3) + config['mcoef_3'] * (rate ** 4)

        return (max(0.0, compute), max(0.0, memory))

        # 部署单个请求
    def place_request(self, request: Request) -> bool:
        # single step result,trying to place the vnf one step
        global result
        # if the request was rejected before the final placment
        for i in range(len(self.request.vtypes)):
            # if self.request.service == 0:
            #     action = self.predict_greedy_memory()
            # elif self.request.service == 1:
            #     action = self.predict_greedy_cpu()
            # elif self.request.service == 2:
            #     action = self.predict_greedy2()
            # else:
            #     action = self.predict_greedy()
            # # action=self.predict_greedy()
            # # placement is done in one step
            self.GRC=GRC(0.85,0.5)
            action=self.GRC.predict(self)
            state, step_reward, result = self.place_single_vnf(action)
            if not result:
                return state, 0.0, result
        # only step_reward is not None the result is True after all the placement
        return state, step_reward, result

    def place_request_datarate(self, request: Request) -> bool:
        # single step result,trying to place the vnf one step
        global result
        # if the request was rejected before the final placment
        for i in range(len(self.request.vtypes)):
            action = self.predict_greedy2()
            # action=self.predict_greedy()
            # placement is done in one step
            state, step_reward, result = self.place_single_vnf(action)
            if not result:
                return state, step_reward, result
        # only step_reward is not None the result is True after all the placement
        return state, step_reward, result

            # place vnf randomly
    def predict_random(self, **kwargs):
        """Samples a valid action from all valid actions."""
        # sample process
        valid_nodes = np.asarray([node for node in self.valid_routes])
        return np.random.choice(valid_nodes)

    def place_vnf(self, node: int) -> bool:
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        if vnum==0:
            compute, memory = self.compute_resources(node, vtype,place=True)
            self.computing[node] -= compute
            # self.memory[node] -= memory
            self.vtype_bidict[(node, vtype)] = self.request
            occupied = {'compute': compute, 'memory': memory, 'datarate': 0.0}
            self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}
            self.routes_bidict[self.request]=(None,node)
            return False
        else:
        # update provisioned compute and memory resources
            compute, memory = self.compute_resources(node, vtype,place=True)
            self.computing[node] -= compute
            # self.memory[node] -= memory

        # track increase of resources occupied by the service deployment
            occupied = {'compute': compute, 'memory': memory, 'datarate': 0.0}
            self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}

            # steer traffic across shortest path (in terms of latency) route
            route = self.valid_routes[node]
            self.steer_traffic(route)

        # update data structure that track relations among VNFs, services and nodes
            self.vtype_bidict[(node, vtype)] = self.request

        # the service is completely deployed; register demanded resources for deletion after duration is exceeded
        if len(self.vtype_bidict.mirror[self.request]) == len(self.request.vtypes):
            return True

        return False

    def steer_traffic(self, route: List) -> None:
        '''Steer traffic from node-to-node across the given route.'''

        for (src, trg) in route:
            # update residual datarate & latency that remains after steering action
            self.datarate[frozenset({src, trg})] -= self.request.datarate
            self.request.resd_lat -= self.propagation[frozenset({src, trg})]

            # register link to routing (link embeddings) of `self.request`
            self.routes_bidict[self.request] = (src, trg)

        # track increase of resources occupied by the service deployment
        datarate = len(route) * self.request.datarate
        occupied = {'compute': 0.0, 'memory': 0.0, 'datarate': datarate}
        # {'compute': 0.24598152161708728, 'memory': 768.0, 'datarate': 516.8745179973857}
        self.occupied = {key: self.occupied[key] + occupied[key] for key in occupied}

    def update_actions(self) -> None:
        '''Update the set of valid placement actions and their respective routings.'''
        # return if simulation episode is already done
        if self.done:
            return

        # compute latencies by shortest path of propagation delays across amenable edges
        # SOURCE:8
        # _, source = self.routes_bidict[self.request][-1]
        # all candidate nodes
        # {8: 0, 11: 0.15430500052566462, 13: 0.3437491777902329, 2: 0.35707748853512133, 31: 0.4124103533704826, 37: 0.5030940551965568, 3: 0.5843501633478713, 25: 0.6353675928592062, 32: 0.6760395849780623, 49: 0.7085530236466248, 41: 0.7594958203022628, 18: 0.8549415707925507, 5: 0.871545738099399, 19: 0.8978817704237891, 34: 0.9212570930293987, 10: 1.0074296241720928, 22: 1.0196577590535458, 16: 1.0266557663712659, 43: 1.0297492850629224, 20: 1.0331334967709591, 40: 1.0457565811621277, 45: 1.0472349789538118, 44: 1.0492127092039008, 1: 1.0590295521644042, 14: 1.0854660739669235, 9: 1.093285762483712, 35: 1.1415510163095033, 12: 1.1603106123610458, 26: 1.1907463656011061, 24: 1.1983889665552838, 48: 1.203296226672668, 33: 1.2113384805366214, 28: 1.2218790047642565, 47: 1.233365330785172, 4: 1.2383230324686378, 29: 1.2506988426677474, 21: 1.2527392953289849, 39: 1.2580420645835757, 6: 1.2771533725283974, 23: 1.3057044419516144, 27: 1.348042876801162, 30: 1.3560566196352284, 38: 1.3872327479467736, 0: 1.3928588300638047, 7: 1.408466492933099, 42: 1.451759545593431, 46: 1.4634045165613112, 15: 1.513905458744372, 17: 1.5145350375183448, 36: 1.608286920926357}
        # 按照权重进行排序，按剩余时长进行截断
        #先获取备选节点,选取两个节点之间的最短路径
        #第一个节点
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        cdemands, mdemands = {}, {}
        if vnum==0:
            for node in self.net.nodes:
                compute, memory = self.compute_resources(node, vtype,place=False)
                cdemands[node] = compute
                mdemands[node] = memory
            #获取第一跳的备选节点
            valid_nodes = [node for node in self.net.nodes if cdemands[node] <=self.computing[node] and mdemands[node]<=self.memory[node]]
            self.valid_routes={node:"the first node" for node in valid_nodes}
        else:
            _, source = self.routes_bidict[self.request][-1]
            # all candidate nodes
            # {8: 0, 11: 0.15430500052566462, 13: 0.3437491777902329, 2: 0.35707748853512133, 31: 0.4124103533704826, 37: 0.5030940551965568, 3: 0.5843501633478713, 25: 0.6353675928592062, 32: 0.6760395849780623, 49: 0.7085530236466248, 41: 0.7594958203022628, 18: 0.8549415707925507, 5: 0.871545738099399, 19: 0.8978817704237891, 34: 0.9212570930293987, 10: 1.0074296241720928, 22: 1.0196577590535458, 16: 1.0266557663712659, 43: 1.0297492850629224, 20: 1.0331334967709591, 40: 1.0457565811621277, 45: 1.0472349789538118, 44: 1.0492127092039008, 1: 1.0590295521644042, 14: 1.0854660739669235, 9: 1.093285762483712, 35: 1.1415510163095033, 12: 1.1603106123610458, 26: 1.1907463656011061, 24: 1.1983889665552838, 48: 1.203296226672668, 33: 1.2113384805366214, 28: 1.2218790047642565, 47: 1.233365330785172, 4: 1.2383230324686378, 29: 1.2506988426677474, 21: 1.2527392953289849, 39: 1.2580420645835757, 6: 1.2771533725283974, 23: 1.3057044419516144, 27: 1.348042876801162, 30: 1.3560566196352284, 38: 1.3872327479467736, 0: 1.3928588300638047, 7: 1.408466492933099, 42: 1.451759545593431, 46: 1.4634045165613112, 15: 1.513905458744372, 17: 1.5145350375183448, 36: 1.608286920926357}
            # 按照权重进行排序，按剩余时长进行截断(迪节斯特拉最短路径)
            lengths, routes = nx.single_source_dijkstra(self.net, source=source, weight=self.get_weights,cutoff=self.request.resd_lat)
            routes = {node: route for node, route in routes.items() if lengths[node] <= self.request.resd_lat}
            vnum = len(self.vtype_bidict.mirror[self.request])
            vtype = self.request.vtypes[vnum]
            cdemands, mdemands = {}, {}
            # the candidate node compute_resources
            for node in routes:
                compute, memory = self.compute_resources(node, vtype,place=False)
                cdemands[node] = compute
                mdemands[node] = memory

            # valid nodes must provision enough compute and memory resources for the deployment
            valid_nodes = [node for node in routes if cdemands[node] <=
                           self.computing[node] and mdemands[node]<=self.memory[node]]
            self.valid_routes = {node: self.get_edges(route) for node, route in routes.items() if node in valid_nodes}
        # lengths, routes = nx.single_source_dijkstra(self.net, source=source, weight=self.get_weights,
        #                                             cutoff=self.request.resd_lat)
        # routes:11:[8,11] candidate paths and all the nodes passed through
        # filter routes to deployment nodes where the routing delay exceeds the maximum end-to-end latency of the request
        # 检查是否有时延未超标的节点,先删减这两条跳之间的路经
        # routes = {node: route for node, route in routes.items() if lengths[node] <= self.request.resd_lat}
        # # check whether reachable nodes also provision enough resources for the deployment
        # vnum = len(self.vtype_bidict.mirror[self.request])
        # vtype = self.request.vtypes[vnum]
        # cdemands, mdemands = {}, {}
        # # the candidate node compute_resources
        # for node in routes:
        #     compute, memory = self.compute_resources(node, vtype)
        #     cdemands[node] = compute
        #     mdemands[node] = memory

        # valid nodes must provision enough compute and memory resources for the deployment
        # valid_nodes = [node for node in routes if cdemands[node] <=
        #                self.computing[node] and mdemands[node] <= self.memory[node]]
        # get the candidate nodes
        # for node in valid_nodes:
        #     # 筛选有终点路由的路径
        #     try:
        #         _, check_route = nx.single_source_dijkstra(self.net, source=node, target=self.request.egress,
        #                                                    weight=self.get_weights,
        #                                                    cutoff=self.request.resd_lat)
        #     except Exception as e:
        #         valid_nodes.pop(valid_nodes.index(node))
        # cache valid routes for the upcoming time step
        # check the valid routes
        # {8: [], 11: [(8, 11)], 13: [(8, 13)], 2: [(8, 2)], 3: [(8, 11), (11, 3)], 31: [(8, 11), (11, 31)], 25: [(8, 13), (13, 25)], 49: [(8, 2), (2, 37), (37, 49)], 37: [(8, 2), (2, 37)], 32: [(8, 11), (11, 31), (31, 32)], 34: [(8, 2), (2, 37), (37, 34)], 41: [(8, 2), (2, 37), (37, 41)], 43: [(8, 11), (11, 3), (3, 43)], 20: [(8, 11), (11, 3), (3, 20)], 10: [(8, 13), (13, 25), (25, 10)], 5: [(8, 11), (11, 31), (31, 32), (32, 5)], 19: [(8, 13), (13, 25), (25, 19)], 18: [(8, 13), (13, 25), (25, 18)], 45: [(8, 2), (2, 37), (37, 49), (49, 45)], 1: [(8, 2), (2, 37), (37, 34), (34, 1)], 40: [(8, 2), (2, 37), (37, 41), (41, 40)], 16: [(8, 13), (13, 25), (25, 19), (19, 16)], 21: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 21)], 22: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22)], 4: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 4)], 44: [(8, 13), (13, 25), (25, 19), (19, 44)], 26: [(8, 2), (2, 37), (37, 34), (34, 26)], 14: [(8, 13), (13, 25), (25, 10), (10, 14)], 35: [(8, 13), (13, 25), (25, 10), (10, 35)], 6: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6)], 39: [(8, 13), (13, 25), (25, 10), (10, 35), (35, 39)], 28: [(8, 13), (13, 25), (25, 19), (19, 44), (44, 28)], 9: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9)], 27: [(8, 11), (11, 3), (3, 43), (43, 27)], 24: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 24)], 47: [(8, 2), (2, 37), (37, 34), (34, 1), (1, 47)], 30: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 30)], 12: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 12)], 48: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 48)], 33: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 33)], 23: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 23)], 29: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 12), (12, 29)], 42: [(8, 13), (13, 25), (25, 19), (19, 16), (16, 9), (9, 23), (23, 42)], 17: [(8, 2), (2, 37), (37, 49), (49, 45), (45, 24), (24, 17)], 0: [(8, 13), (13, 25), (25, 10), (10, 14), (14, 48), (48, 0)], 46: [(8, 13), (13, 25), (25, 19), (19, 44), (44, 28), (28, 46)], 38: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 38)], 7: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 7)], 15: [(8, 11), (11, 3), (3, 43), (43, 27), (27, 15)], 36: [(8, 11), (11, 31), (31, 32), (32, 5), (5, 22), (22, 6), (6, 38), (38, 36)]}
        # every step's valid_routes
        # self.valid_routes = {node: ServiceCoordination7.get_edges(route) for node, route in routes.items() if
        #                      node in valid_nodes}
    def rate_init(self):
        self.rate_list=[]
        for node_id in range(len(list(self.net.nodes))):
            self.rate_list.append([])
            for i in range(6):
                self.rate_list[node_id].append([0.0,0.0,0.0,0.0])

    def update_time_window(self):
        #rejected_request in the last timestep
        # self.release_request()
        self.start_time += self.TIME_WINDOW_LENGTH
        self.end_time += self.TIME_WINDOW_LENGTH
        # update the self.time,we should release the request according to the length of all queues
        self.time = self.end_time
        # self.compute_network_util()
        # self.compute_network_util()
        # self.compute_network_util()
        # self.reconfiguration()
        self.time_window(self.start_time, self.end_time)
    #只有这一点点用了
    def update_interval_time(self):
        # 直视我栽种,你的time为什么不对
        length = len(self.all_waiting_queues)
        self.update_interval = self.TIME_WINDOW_LENGTH / (length)

    def compute_state_admission(self):
        state_compute=[]
        for node in self.net.nodes:
            state_compute.append(self.rate_list[node][0][3]/40)
            state_compute.append(self.rate_list[node][2][3]/40)
            other_total_rate=0
            state_compute.append((self.rate_list[node][0][0]+self.rate_list[node][0][1]+self.rate_list[node][0][2])/40)
            state_compute.append((self.rate_list[node][2][0]+self.rate_list[node][2][1]+self.rate_list[node][2][2])/40)
            # for vtype in self.rate_list[node]:
            #     if self.rate_list[node].index(vtype)==0 or self.rate_list[node].index(vtype)==2:
            #         for i in range(3):
            #             other_total_rate+=vtype[i]
            #         state_compute.append(other_total_rate/(40))
            # state_compute.append(self.computing[node])
        # state_network=[vtype for node in self.net.nodes for vtype in self.rate_list[node]]
        state_network=[self.datarate[frozenset(
            {src, trg})] / self.MAX_LINKRATE for src, trg in self.net.edges]
        # accept_rate_list = self.calculate_accept_rate()
        queue_datarate=self.calculate_queue_state()
        return np.asarray(list(chain(state_compute,state_network,queue_datarate,[self.start_time])))
    def calculate_queue_state(self):
        return [len(self.all_waiting_queues)]
    def release_request(self):
        while self.deployed and self.deployed[0][0] < self.end_time:
            self.deployed.sort(key=lambda s: s[0])
            try:
                rel_time, service = self.deployed.pop(0)
            except:
                print(self.deployed)
                print(self.time)
            self.release2(service)

    def initial_one_request(self, request: Request):
        # self.routes_bidict[request] = (None, request.ingress)
        # self.info[request.service].requests += 1
        self.num_requests+=1
        request.schedule_time=self.end_time
        # set requested VNFs upon arrival of request
        request.resd_lat = request.max_latency
        # set the profit for the request
        request.vtypes = self.services[request.service]

    # def initial_one_rejected_request(self, request: Request):
    #     if self.routes_bidict[request] == []:
    #         self.routes_bidict[request] = (None, request.ingress)

    def calculate_accept_rate(self):
        accept_rate = []
        for i in range(4):
            if (self.info[i].requests)!=0:
                accept_rate_single = self.info[i].accepts / (self.info[i].requests)
                accept_rate.append(accept_rate_single)
            else:
                accept_rate.append(0.0)
        for i in range(4):
            if accept_rate[i]>1.0:
                # print("接受个数"+str(self.info[i].accepts))
                # print("请求个数"+str(self.info[i].requests))
                raise ValueError("接受率有问题")
        if self.time==45:
            for i in range(4):
                # print("接受个数" + str(self.info[i].accepts))
                # print("请求个数" + str(self.info[i].requests))
                # time.sleep(1)
                print("本论结束")
                print("接受个数" + str(self.info[i].accepts))
                print("请求个数" + str(self.info[i].requests))
        return accept_rate

    def valid_queues(self,action):
        if self.all_waiting_queues[action] == []:
            return False
        return True

    def valid_actions_method(self):
        valid_actions= {}
        for queue in self.all_waiting_queues:
            if queue != []:
                valid_actions[self.all_waiting_queues.index(queue)]=len(queue)
        if self.needed_updating_time_window():
            # print("直视我,栽种")
            # print(self.all_waiting_queues)
            # print(self.time)
            self.update_time_window()
            # self.update_interval_time()
            for queue in self.all_waiting_queues:
                if queue != []:
                    valid_actions[self.all_waiting_queues.index(queue)] = len(queue)
        return valid_actions


    def needed_updating_time_window(self):
        #这跟队列开的速率和离开速率有关
        if self.check_time_window_queue()==1 and self.end_time<self.TIME_HORIZON and ((self.time % self.TIME_WINDOW_LENGTH<1e-14) or (self.end_time+self.TIME_WINDOW_LENGTH-self.time<1e-13)) and self.time!=self.TIME_WINDOW_LENGTH and self.end_time!=self.TIME_HORIZON:
            # print("队列全空时，等待下一次更新")
            return True
        else:
            return False


    def predict_greedy(self,  **kwargs):
        valid_actions = self.valid_routes.keys()
        _, pnode = self.routes_bidict[self.request][-1]

        # compute delays (including processing delays) associated with actions
        # 计算与操作相关的延迟（包括处理延迟）
        lengths, _ = nx.single_source_dijkstra(self.net, source=pnode, weight=self.get_weights,
                                               cutoff=self.request.resd_lat)

        # choose (valid) VNF placement with min. latency increase
        action = min(valid_actions, key=lengths.get)

        return action

    def predict_greedy2(self, **kwargs):
        valid_actions = self.valid_routes.keys()
        _, pnode = self.routes_bidict[self.request][-1]

            # compute delays (including processing delays) associated with actions
            # 计算与操作相关的延迟（包括处理延迟）
        lengths, _ = nx.single_source_dijkstra(self.net, source=pnode, weight=self.get_weights2)

            # choose (valid) VNF placement with min. latency increase
        # # print(lengths.get)
        action = max(valid_actions, key=lengths.get)
        return action

        # calculate the network temperature
    def compute_network_util(self):
        node_util_list=[]
        link_util_list=[]
        for u, d in self.net.nodes(data=True):
            node_util=self.computing[u]/d['compute']
            mem_util=self.memory[u]/d['memory']
            node_util_list.append((node_util,mem_util))
        print("节点利用率")
        print(node_util_list)
        for u, v, w in self.net.edges(data=True):
            link_util=self.datarate[frozenset({u, v})]/(w['datarate'])
            propagation=self.propagation[frozenset({u, v})]/w['propagation']
            link_util_list.append((u,v,link_util,propagation))
        print("链路利用率")
        print(link_util_list)
        return node_util_list,link_util_list

    #TODO:利用网络温度作为重配置的标准
    def compute_compute_temperature(self):
        c_sum=0
        m_sum=0
        node_util,link_util=self.compute_network_util()
        #not the first time window
        if self.end_time!=self.TIME_WINDOW_LENGTH:
            #t 时刻的cpu利用率总和
            for c_util,m_util in node_util:
                #计算当前时间窗口
                c_sum+=c_util
                m_sum+=m_util
            #计算上一个时间窗口整个网络的平均利用率之和(算分子)
            self.last_compute_state = self.compute_state_list[-1]
            #加入当前队列
            self.compute_state_list.append(c_sum)
            #计算当前窗口和之前窗口差值
            delta_compute=c_sum-self.last_compute_state
            #同样的计算内存
            self.last_memory_state = self.memory_state_list[-1]
            self.memory_state_list.append(c_sum)
            delta_memory = m_sum - self.last_memory_state
            entroy_compute_sum=0
            entroy_memory_sum=0
            #计算瑞丽熵，选取指标是利用率两点的空间分布
            #Todo：加上时序中间过程采样点上的数据
            for c_util,m_util in node_util:
                c_prob=c_util/c_sum
                log_c_prob=np.log(c_prob)
                entroy_compute=c_prob*log_c_prob
                #计算计算瑞丽熵综总和
                entroy_compute_sum += entroy_compute
                m_prob=m_util/m_sum
                log_m_prob=np.log(m_prob)
                entroy_memory = m_prob * log_m_prob
                #计算内存瑞丽熵总和
                entroy_memory_sum += entroy_memory
            #计算两个窗口端点
            #计算上一个时间窗口开始时的时候的计算瑞丽熵值(算分母)
            self.last_compute_entory= self.entroy_compute_list[-1]
            #当前时间的计算熵
            self.entroy_compute_list.append(entroy_compute_sum)
            delta_entroy_compute=entroy_compute_sum-self.last_compute_entory
            #计算温度的公式
            compute_temperature=abs(delta_compute)/abs(delta_entroy_compute)
            self.last_memory_entroy = self.entroy_memory_list[-1]
            self.entroy_memory_list.append(entroy_memory_sum)
            delta_entroy_memory = entroy_memory_sum - self.last_memory_entroy
            memory_temperature=abs(delta_memory)/abs(delta_entroy_memory)
            # print("当前计算温度"+str(compute_temperature))
            # print("当前存储温度"+str(memory_temperature))
            return compute_temperature,memory_temperature


    def init_entroy(self):
        node_util, link_util = self.compute_network_util()
        c_sum = 0
        m_sum = 0
        entroy_compute_sum=0
        entroy_memory_sum=0
        for c_util, m_util in node_util:
            # 计算当前时间窗口
            c_sum += c_util
            m_sum += m_util
        for c_util, m_util in node_util:
            c_prob = c_util / c_sum
            log_c_prob = np.log(c_prob)
            entroy_compute = c_prob * log_c_prob
            # 计算计算瑞丽熵综总和
            entroy_compute_sum += entroy_compute
            m_prob = m_util / m_sum
            log_m_prob = np.log(m_prob)
            entroy_memory = m_prob * log_m_prob
            # 计算内存瑞丽熵总和
            entroy_memory_sum += entroy_memory
        self.compute_state_list=[0]
        self.memory_state_list=[0]
        self.entroy_compute_list=[entroy_compute_sum]
        self.entroy_memory_list = [entroy_memory_sum]
        return entroy_compute_sum,entroy_memory_sum


    def check_valid_routes(self,request : Request):
        _, route = nx.single_source_dijkstra(self.net, source=request.ingress, target=request.egress,cutoff=request.max_latency)


    def predict_greedy_cpu(self, **kwargs):
        """Samples a valid action from all valid actions."""
        # sample process
        valid_nodes = np.asarray([node for node in self.valid_routes])
        valid_nodes_cpu_dict={node:self.computing[node] for node in self.valid_routes}
        sort_result=sorted(valid_nodes_cpu_dict.items(), key=lambda d: d[1],reverse=True)
        return sort_result[0][0]

    def predict_greedy_memory(self, **kwargs):
        """Samples a valid action from all valid actions."""
        # sample process
        valid_nodes = np.asarray([node for node in self.valid_routes])
        valid_nodes_memory_dict={node:self.memory[node] for node in self.valid_routes}
        sort_result=sorted(valid_nodes_memory_dict.items(), key=lambda d: d[1],reverse=True)
        return sort_result[0][0]

    def predict_greedy_memory_cpu(self, **kwargs):
        """Samples a valid action from all valid actions."""
        # sample process
        valid_nodes = np.asarray([node for node in self.valid_routes])
        valid_nodes_memory_dict={node:self.memory[node]*self.computing[node] for node in self.valid_routes}
        sort_result=sorted(valid_nodes_memory_dict.items(), key=lambda d: d[1],reverse=True)
        return sort_result[0][0]

    def takeSecond(elem):
        return elem.datarate

                # self.compute_compute_temperature()
        # sim_env=copy.deepcopy(self)
        # #遍历循环,看看能不能换个请求方式
        # list_cpu=sim_env.get_candidate_requests_cpu()
        # print("备选请求")
        # print(list_cpu)
        # if list_cpu!=[]:
        #     # listed_cpu = list_cpu.sort(key=lamda list_cpu:list_cpu.datarate)
        #     sorted(list_cpu,key=lambda request:request.datarate,reverse=True)
        #     sim_env.request=list_cpu[0]
        #     print("重部署之前的状态")
        #     print(sim_env.vtype_bidict.mirror[sim_env.request])
        #     sim_env.release2(sim_env.request)
        #     # print("开始重新部署节点")
        #     sim_env.initial_one_rejected_request(sim_env.request)
        #     sim_env.update_actions()
        #     sim_env.compute_state()
        #     resultcpu = False
        #     try:
        #         state_placement, reward_step, resultcpu = sim_env.place_request_cpu(sim_env.request)
        #         print("重部署之后的状态")
        #         print(sim_env.vtype_bidict.mirror[sim_env.request])
        #     except Exception as e:
        #         pass
        #     if resultcpu:
        #     #对真实环境开始重配置
        #         list_cpu_real = self.get_candidate_requests_cpu()
        #         for req in list_cpu_real :
        #             if req.ingress == sim_env.request.ingress and req.egress == sim_env.request.egress and sim_env.request.max_latency==req.max_latency:
        #                 self.request = list_cpu_real [list_cpu_real.index(req)]
        #         self.release2(self.request)
        #         self.initial_one_rejected_request(request=self.request)
        #         self.update_actions()
        #         self.compute_state()
        #         self.info[self.request.service].accepts -=1
        #         state_placement, reward_step, result = self.place_request_cpu(self.request)

    def get_candidate_requests_cpu(self):
        node_util, link_util = self.compute_network_util()
        cpu_util = []
        mem_util = []
        for node in node_util:
            cpu_util.append(node[0])
            mem_util.append(node[1])
        #选择CPU利用率最大的节点
        sorted_cpu = sorted(enumerate(self.computing), key=lambda x: x[1])
        idx1 = [i[0] for i in sorted_cpu]
        nums1 = [i[1] for i in sorted_cpu]
        #CPU利用率最大的节点，取他上面的1，3，5进行迁移,所有待迁移请求
        set_cpu=set()
        request_cpu_3=[]
        for i in  range(6):
            for req  in self.vtype_bidict[(sorted_cpu[0][0],i)]:
                set_cpu.add(req)
        # if request_cpu_3!=[]:
        #     time.sleep(2)
        # request_cpu_1=request_cpu_3
        # set_cpu=set()
        # for req in request_cpu_1:
        #     set_cpu.add(req)
        list_cpu=list(set_cpu)
        return list_cpu

    #TDOD:RECONFIGURE THE NETWORK
    def get_candidate_requests_route_link(self):
        #获取当前最拥堵的底层链路
        node_util, link_util = self.compute_network_util()
        link_util1=[]
        for datarate in link_util:
            link_util1.append(datarate[2])
        sorted_link = sorted(enumerate(self.datarate), key=lambda x: x[1])
        idx1 = [i[0] for i in sorted_link]
        nums1 = [i[1] for i in sorted_link]
        set_link=set()
        # print(link_util)
        for req in self.routes_bidict.mirror[(link_util[idx1[0]][0],link_util[idx1[0]][1])]:
            set_link.add(req)
        list_link=list(set_link)
        return list_link

    def check_time_window_queue(self):
        # flag=0
        # for queue in self.all_waiting_queues:
        #     if len(queue) == 0:
        #         flag+=1
        #     else:
        #         flag=flag
        # return flag
        if self.all_waiting_queues != []:
            return False
        else:
            return True

    def check_end(self):
        if self.all_waiting_queues==[] and self.end_time==45:
            self.done=True
            return True
        elif self.all_waiting_queues==[]:
            self.update_time_window()
            return False
        else:
            return False
    #0-1判断
    def step(self, action):
        # updating the request,all the request has initalized in the updating_time_window
        # # print("当前队列状态")
        # # print(self.all_waiting_queues)
        #所有队列都空了，直接拒绝，等风来
        #我的队列不可能空的,你到头了,明白吗?
        self.all_requests+=1
        if (isinstance(action, tuple)):
            action = action[0]
        else:
            action = int(action)
        # if len(self.all_waiting_queues)==0:
        #     self.done=True
        #     return self.compute_state_admission(), 0.0, self.done, {}
        if action==0:
            # if self.all_waiting_queues==[] and self.check_end()==False:
            #     self.update_time_window()
            self.request = self.all_waiting_queues[0]
            self.info[self.request.service].requests += 1
            del self.all_waiting_queues[self.all_waiting_queues.index(self.request)]
            self.info[self.request.service].num_rejects+=1
            # self.time = self.time + self.update_interval
            # self.valid_actions = self.valid_actions_method()
            # self.release_request()
            # time.sleep(0.5)
            # print(self.time)
            # result_update = self.check_time_window_queue()
            # if result_update:
            #     self.update_time_window()
            #     self.update_interval_time()
            #     print("更新后的队列")
            #     print(self.all_waiting_queues)
            self.check_end()
            return self.compute_state_admission(), 0.0, self.done, {}
        else:
        #有请求
            self.request = self.all_waiting_queues[0]
            del self.all_waiting_queues[0]
            # print(self.all_waiting_queues)
            #大家都是新请求,玩你麻痹聊斋呢
            self.info[self.request.service].requests += 1
            #because i removed the self.routes_bidict and self.vtype_bidict,so i should to reinitial it
            self.update_actions()
            #because a request is removed by it can't access the endpoint because of there is no route
            if not self.valid_routes:
                # print("服务由于没有到终点的路由一开始被拒绝"+str(self.request.service))
                # self.time=self.time+self.update_interval
                # self.valid_actions = self.valid_actions_method()
                # self.release_request()
                # time.sleep(0.5)
                # print(self.time)
                self.check_end()
                return self.compute_state_admission(), 0.0, self.done, {}
            # whether this place is right or not currently i should ignore it
            self.compute_state()
            state_placement, reward_step, result = self.place_request(self.request)
            # the request is accepted
            if not result:
                reward = reward_step
            # the request is rejected.However it should be send to the reject queue accroding to the servece type
            else:
                reward = reward_step
            # self.time = self.time + self.update_interval
            # self.valid_actions=self.valid_actions_method()
            # the time_window has  finished yet
            # the time_window has not finished we should update the time
            # print(self.time)
            # self.release_request()
            # result_update=self.check_time_window_queue()
            # if result_update:
            #     # self.update_time_window()
            #     # # self.update_interval_time()
            #     # print("接受更新后的队列")
            #     # print(self.all_waiting_queues)
            self.check_end()
            return self.compute_state_admission(), reward, self.done, {}

    @staticmethod
    def score2(rate_list, config, request):
        '''Score the CPU and memory resource consumption for a given VNF configuration and requested datarate.'''
        # set VNF resource consumption to zero whenever their requested rate is zero
        for rate in rate_list:
            if rate < 0.0:
                rate_list[rate_list.index(rate)]=0.0
        compute = ceil((rate_list[0] + rate_list[1] + rate_list[2]) / (config['max. req_transf_rate']))* config['coff'] + \
                      config['ccoef_1'] * (rate_list[0] +  \
                      (rate_list[1]) + \
                      (rate_list[2])+(rate_list[3]))
        memory = ceil((rate_list[0] + rate_list[1] + rate_list[2]) / (config['max. req_transf_rate'])) * config[
                'moff'] + \
                      config['mcoef_1'] * (rate_list[0] + \
                                           (rate_list[1]) + \
                                           (rate_list[2])+(rate_list[3]))
        return compute,memory
        # if request.service != 3:
        #     compute = ceil((rate_list[0] + rate_list[1] + rate_list[2]) / (config['max. req_transf_rate']))* config['coff'] + \
        #               config['ccoef_1'] * (rate_list[0] +  \
        #               (rate_list[1]) + \
        #               (rate_list[2]))
        #     memory = ceil((rate_list[0] + rate_list[1] + rate_list[2]) / (config['max. req_transf_rate'])) * config[
        #         'moff'] + \
        #               config['mcoef_1'] * (rate_list[0] + \
        #                                    (rate_list[1]) + \
        #                                    (rate_list[2]))
        #     return compute,memory
        # else:
        #     compute = ceil((rate_list[3]) / (80) )* config['coff'] + config['ccoef_1'] * \
        #               rate_list[3]
        #     memory = ceil((rate_list[3]) / (80)) * config['moff'] + config['mcoef_1'] * rate_list[3]

    def compute_rate_release(self,node,vtype,config,request):
        supplied_rate = sum([service.datarate for service in self.vtype_bidict[(node, vtype)]])
        prev_cdem, prev_mdem = self.score(supplied_rate, config)
        after_cdem, after_mdem = self.score(supplied_rate - request.datarate, config)
        return prev_cdem, prev_mdem, after_cdem, after_mdem
    def compute_rate_demand(self,node,vtype,config,request):
        supplied_rate = sum(
            [service.datarate for service in self.vtype_bidict[(node, vtype)]])
        prev_cdem, prev_mdem = self.score(supplied_rate, config)
        after_cdem, after_mdem = self.score(supplied_rate + request.datarate, config)
        return prev_cdem, prev_mdem, after_cdem, after_mdem
    def compute_rate_release_2(self,node,vtype,config,request,place):
        # for service in self.vtype_bidict[(node, vtype)]:
        #     self.rate_list[node][vtype][service.service]+=service.datarate
        prev_cdem, prev_mdem = self.score2(self.rate_list[node][vtype], config,request)
        self.rate_list[node][vtype][request.service]=self.rate_list[node][vtype][request.service] - request.datarate
        after_cdem, after_mdem = self.score2(self.rate_list[node][vtype], config,request)
        if not place:
            self.rate_list[node][vtype][request.service] = self.rate_list[node][vtype][request.service] + request.datarate
        return prev_cdem,prev_mdem,after_cdem,after_mdem
    def compute_rate_demand_2(self,node,vtype,config,request,place):
        # for service in self.vtype_bidict[(node, vtype)]:
        #     self.rate_list[node][vtype][service.service]+=service.datarate
        prev_cdem, prev_mdem = self.score2(self.rate_list[node][vtype], config,request)
        self.rate_list[node][vtype][request.service] = self.rate_list[node][vtype][request.service] + request.datarate
        after_cdem, after_mdem = self.score2(self.rate_list[node][vtype], config,request)
        if not place:
            self.rate_list[node][vtype][request.service] = self.rate_list[node][vtype][request.service] - request.datarate
        return prev_cdem,prev_mdem,after_cdem,after_mdem

    def choose_valid_actions(self):
        valid_actions=[1,1]
        return np.array(valid_actions)