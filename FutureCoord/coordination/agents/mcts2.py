import time
from copy import deepcopy
from typing import Dict, List, Tuple
from abc import abstractmethod
from more_itertools import peekable

import numpy as np
from pairing_functions import cantor

from coordination.agents.baselines import RandomPolicy
from coordination.environment.traffic import Traffic
# from coordination.environment.admission_control import ServiceCoordination5


class ANode:
    def __init__(self, parent, action):
        self.parent: ANode = parent
        self.action: int = action
        self.children: List[ANode] = []
        self.visits: int = 0
        self.avalue: float = 0.0


class MCTS:
    def __init__(self, C: float, max_searches: int, seed: int = None, **kwargs: Dict) -> None:
        '''Initialize Monte Carlo tree search algorithm.'''
        self.C = C
        self.rng = np.random.default_rng(seed)
        self.max_searches = max_searches
        self.rpolicy=RandomPolicy(seed=seed)

    def learn(self, **kwargs: Dict) -> None:
        '''When a non-default rollout policy is applied, call the training hook first.'''
        pass

    def reset(self) -> None:
        '''Reset grown search tree.'''
        self.root = None

    @abstractmethod
    def evaluate(self, sim_env) -> float:
        '''Heuristic evaluation function to assess node value.'''
        pass

    def select_and_expand(self, sim_env) -> Tuple:
        '''Traverse the search tree downwards and expand it by another unvisted node.'''
        node = self.root
        rewards = []
        #我带过来的一定要是一个是实例化好的的请求
        while not sim_env.done1:
            valid = [action for action in sim_env.valid_routes]
            # enable REJECT_ACTION only upon the arrival of a novel request
            #我不拒绝任何请求
            # if not sim_env.request in sim_env.vtype_bidict.mirror:
            #     valid += [sim_env.REJECT_ACTION]

            valid_visited = [child for child in node.children if child.action in valid]
            # case: not all valid actions have been visited before; select among unexplored nodes
            if len(valid_visited) < len(valid):
                choices = set(valid) - set(child.action for child in valid_visited)
                action = self.rng.choice(list(choices))
                child = ANode(node, action)
                node.children.append(child)
                sim_env.place_single_vnf(action)
                reward = self.evaluate(sim_env)
                rewards.append(reward)
                return child, rewards

            # case: all valid actions have been visited before; progress in search tree
            avalues = np.asarray([child.avalue for child in valid_visited])
            visits = np.asarray([child.visits for child in valid_visited])

            # choose child node according to UCT formula
            uct = avalues +self.C * np.sqrt((2 * np.log(np.sum(visits))) / visits)
            choice = np.argmax(uct)
            node = valid_visited[choice]

            # update simulation environment and search; proceed in search tree
            sim_env.place_single_vnf(node.action)
            reward = self.evaluate(sim_env)
            rewards.append(reward)
        # reached end-of-episode
        return node, rewards

    def backpropagate(self, node: ANode, rewards: List) -> None:
        '''Backpropagate simulated rewards along the traversed search tree.'''
        rewards = np.cumsum(rewards[:: -1])
        for rew in rewards:
            node.visits += 1
            node.avalue = ((node.visits - 1) / node.visits) * node.avalue + (rew / node.visits)
            node = node.parent

    def rollout(self, sim_env) -> float:
        '''Obtain a Monte Carlo return estimate by simulating the episode until termination.'''
        mc_return = 0.0
        # simulate rollout phase until termination
        # obs=sim_env.compute_state()
        while not sim_env.done1:
            # use default (rollout) policy to select an action
            action = self.rpolicy.predict(observation=sim_env.compute_state(), deterministic=True, env=sim_env, process=None)
            obs, reward, _= sim_env.place_single_vnf(action)

            # cumulate simulated rewards for Monte Carlo return estimate
            reward = self.evaluate(sim_env)
            mc_return += reward

        return mc_return

    @abstractmethod
    def flow_forecast(self, sim_env, trace: List) -> List:
        '''Prepare forecast flows for simulating service coordination.'''
        pass

    def predict(self, env, process: Traffic, **kwargs: Dict):
        '''Use Monte Carlo tree search to select the next service deployment decision.'''
        # initialize planning procedure for current simulation environment
        self.root = ANode(None, None)
        sim_env = deepcopy(env)
        valid = [action for action in sim_env.valid_routes]
        if len(valid)==1:
            return valid[0]
        for i in range(1):
            for _ in range(self.max_searches):
            # setup simulation environment for MCTS
                sim_env = deepcopy(env)
                sim_env.planning_mode = True

            # replace future flow arrivals either with forecast flows or with no later arrivals
            #对于mavens,就是当前请求
                sim_env.tempoary_request_list = self.flow_forecast(sim_env, process.sample())
                # select an unexplored node that is a descendent of known node
                node, rewards = self.select_and_expand(sim_env)
                # use rollout policy if no value function is provided
                rol_return = self.rollout(sim_env)
                rewards[-1] += rol_return
                self.backpropagate(node, rewards)
            if np.std([child.visits for child in self.root.children])>=3.0:
                break
        # select child of root node that was visited most
        # # choose child node according to UCT formula
        child = np.argmax([child.visits for child in self.root.children])
        return self.root.children[child].action
        # if np.std([child.visits for child in self.root.children])!=0:
        #     child=np.argmax([child.visits for child in self.root.children])
        #     return self.root.children[child].action
        # else:
        #     return valid[0]
        return self.root.children[child].action

class MavenS(MCTS):

    def __init__(self, C: float, max_searches: int, greediness: float, coefficients: Tuple, seed: int = None,
                 **kwargs: Dict):
        super().__init__(C, max_searches, seed)
        self.greediness = greediness
        self.alpha, self.beta, self.gamma = coefficients
        self.rpolicy = RandomPolicy(seed=seed)

    def flow_forecast(self, sim_env, trace: List) -> List:
        '''MavenS does not consider future flow arrivals.'''
        return []

    def evaluate(self, sim_env) -> float:
        '''Calculate deployment costs from occupied compute, memory and datarate resources upon admission.'''

        # case: intermediate deployment decision
        # if not sim_env.admission['finalized']:
        #     return 0.0
        #
        # # case: to-be-deployed request was rejected
        # elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
        #     return self.greediness
        #
        # # case: to-be-deployed request was successfully deployed
        # compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        # memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        # datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        #
        # # compute reward as resource cost given by weighted (normalized) increase of resources
        # return (-1) * (self.alpha * compute + self.beta * memory + self.gamma * datarate)
        if not sim_env.admission['finalized']:
            return 0.0

        # case: to-be-deployed request was rejected
        elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
            return self.greediness

        # case: to-be-deployed request was successfully deployed
        compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        # print("+++++++++++++")
        # print("service type"+str(sim_env.request.service))
        # print(compute)
        # print(memory)
        # print(datarate)
        # print("++++++++++++++++++")
        # print("service type"+str(sim_env.request.service))
        # print("service compute" + str(compute))
        # print("service memory" + str(memory))
        # print("service datarate" + str(datarate))
        # compute reward as resource cost given by weighted (normalized) increase of resources
        # if entroy_change_memory>0 and sim_env.request.service==1:
        #     return (-1) * (self.alpha * compute+ self.beta * memory/(entroy_change_memory) + self.gamma * datarate)
        # elif entroy_change_compute>0 and sim_env.request.service==0:
        #     return (-1) * (self.alpha * compute/((entroy_change_compute)) + self.beta * memory  + self.gamma * datarate)
        # return (-1/2000) * (self.alpha * compute + self.gamma * datarate)/(abs(entroy_change_compute)+abs(entroy_change_network))
        return (-1) * (self.alpha * compute  + self.gamma * datarate)
        # return (-1) * (self.alpha * compute/abs(entroy_change_compute) + self.beta * memory/abs(entroy_change_memory) + self.gamma * datarate/abs(entroy_change_network))

class FutureCoord(MCTS):
    def __init__(self, C: float, max_searches: int, max_requests: int,  seed: int = None, **kwargs: Dict):
        super().__init__(C, max_searches, seed)
        self.max_requests = max_requests
        self.rpolicy=RandomPolicy(seed=seed)
    def learn(self, **kwargs):
        '''Train default policy if `self.rpolicy` is trainable.'''
        if callable(getattr(self.rpolicy, 'learn', None)):
            self.rpolicy.learn(**kwargs)

    def flow_forecast(self, sim_env, trace: List) -> List:
        '''Simulate `max_requests` forecast flows during the rollout phase.'''
        # only simulate requests that arrive after the to-be-deployed request
        trace = [req for req in trace if sim_env.time < req.arrival]

        # simulate at maximum `max_requests` requests
        trace = trace[: self.max_requests]
        return trace

    def evaluate(self, sim_env) -> float:
        # return float(sim_env.admission['deployed'])
        # case: intermediate deployment decision
        # if not sim_env.admission['finalized']:
        #     return 0.0
        #
        # # case: to-be-deployed request was rejected
        # elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
        #     return self.greediness
        #
        # # case: to-be-deployed request was successfully deployed
        # compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        # memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        # datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        #
        # # compute reward as resource cost given by weighted (normalized) increase of resources
        # return (-1) * (self.alpha * compute + self.beta * memory + self.gamma * datarate)
        if not sim_env.admission['finalized']:
            return 0.0

        # case: to-be-deployed request was rejected
        elif sim_env.admission['finalized'] and not sim_env.admission['deployed']:
            return -5.0

        # case: to-be-deployed request was successfully deployed
        compute = sim_env.occupied['compute'] / max(c for _, c in sim_env.net.nodes('compute'))
        memory = sim_env.occupied['memory'] / max(c for _, c in sim_env.net.nodes('memory'))
        datarate = sim_env.occupied['datarate'] / max(data['datarate'] for _, _, data in sim_env.net.edges(data=True))
        # print("+++++++++++++")
        # print("service type"+str(sim_env.request.service))
        # print(compute)
        # print(memory)
        # print(datarate)
        # print("++++++++++++++++++")
        # print("service type"+str(sim_env.request.service))
        # print("service compute" + str(compute))
        # print("service memory" + str(memory))
        # print("service datarate" + str(datarate))
        # compute reward as resource cost given by weighted (normalized) increase of resources
        # if entroy_change_memory>0 and sim_env.request.service==1:
        #     return (-1) * (self.alpha * compute+ self.beta * memory/(entroy_change_memory) + self.gamma * datarate)
        # elif entroy_change_compute>0 and sim_env.request.service==0:
        #     return (-1) * (self.alpha * compute/((entroy_change_compute)) + self.beta * memory  + self.gamma * datarate)
        # return (-1/2000) * (self.alpha * compute + self.gamma * datarate)/(abs(entroy_change_compute)+abs(entroy_change_network))
        return (-1) * (1 * compute+ 1 * datarate)

    def select_and_expand(self, sim_env) -> Tuple:
        '''Traverse the search tree downwards and expand it by another unvisted node. Inserts delimiter nodes upon arrivals.'''
        node = self.root
        rewards = []

        # track number of processed requests to determine whether new request has arrived
        num_requests = sim_env.num_requests

        while not sim_env.done1:
            # case: next action relates to deployment of NEXT request
            if num_requests < sim_env.num_requests:
                # check whether another request with the same ingress has been simulated before
                pseudo_action = cantor.pair(sim_env.request.ingress, sim_env.request.service, sim_env.request.egress)

                # case: requested service arrives at prev. unseen ingress; insert fitting pseudo node
                visited = [child.action for child in node.children]

                if not pseudo_action in visited:
                    child = ANode(node, pseudo_action)
                    node.children.append(child)
                    rewards.append(0.0)
                    return child, rewards

                # case: fitting pseudo node is already in search tree; proceed selection
                node = next(child for child in node.children if child.action == pseudo_action)
                rewards.append(0.0)

                # update number of simulated service requests
                num_requests = sim_env.num_requests

            # case: next action extends deployment of (real) to-be-deployed request
            else:
                valid = [action for action in sim_env.valid_routes]
                # enable REJECT_ACTION only upon the arrival of a novel request
                # if not sim_env.request in sim_env.vtype_bidict.mirror:
                #     valid += [sim_env.REJECT_ACTION]
                valid_visited = [child for child in node.children if child.action in valid]
                # case: not all valid actions have been visited before; select among missing nodes
                if len(valid_visited) < len(valid):
                    choices = set(valid) - set(child.action for child in valid_visited)
                    action = int(self.rng.choice(list(choices)))
                    child = ANode(node, action)
                    node.children.append(child)

                    sim_env.place_single_vnf(action)
                    reward = self.evaluate(sim_env)
                    rewards.append(reward)
                    return child, rewards

                # case: all valid actions have been visited before; progress in search tree
                avalues = np.asarray([child.avalue for child in valid_visited])
                visits = np.asarray([child.visits for child in valid_visited])

                # choose child node according to UCT formula
                uct = avalues + self.C * np.sqrt((2 * np.log(np.sum(visits))) / visits)
                choice = np.argmax(uct)
                node = valid_visited[choice]

                # update simulation environment and search; proceed in search tree
                sim_env.place_single_vnf(node.action)
                reward = self.evaluate(sim_env)
                rewards.append(reward)
        # reached end-of-episode
        return node, rewards