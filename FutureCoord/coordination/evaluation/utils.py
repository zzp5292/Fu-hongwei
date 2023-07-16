import re
import os

import gym
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from munch import unmunchify
from timeit import default_timer as timer

from sb3_contrib.common.wrappers import ActionMasker
from script2 import NUM_DAYS
from coordination.agents_ac import baselines
from coordination.environment.deployment import ServiceCoordination
from coordination.environment.deployment2 import ServiceCoordination2
from coordination.environment.traffic2 import ServiceTraffic, Traffic, TrafficStub
from stable_baselines3.common.env_util import make_vec_env

from coordination.environment.deployment3 import ServiceCoordination3


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.choose_valid_actions()
def setup_process(rng, exp, services, eday, sdays, load, rate, latency,arrival_rate):
    exp = deepcopy(exp)
    services = deepcopy(services)

    # load endpoint probability matrix used to sample (ingress, egress) tuples for requested services
    # with open(exp.endpoints, 'rb') as file:
    #     endpoints = np.load(file)
    #     endpoints = endpoints[eday]

    # compute shortest paths in terms of propagation delays to define max. end-to-end latencies
    G = nx.read_gpickle(exp.overlay)
    spaths = dict(nx.all_pairs_dijkstra_path_length(G, weight='propagation'))

    processes = []
    for snum, (service, day) in enumerate(zip(services, sdays)):
        traffic = service.process.marrival
        with open(traffic, 'rb') as file:
            traffic = np.load(file)
            # scale arrival rates according to const. load factor
            traffic = load * traffic[day]

        # scale mean of distributions by scaling factor (usually set to 1.0)
        service.datarates.loc = service.datarates.loc * rate 
        service.latencies.loc = service.latencies.loc * latency 

        process = ServiceTraffic(rng, snum, exp.time_horizon, service.process, service.datarates, service.latencies,traffic, spaths,arrival_rate)
        processes.append(process)

    return Traffic(processes)

def setup_agent(config, env, seed):
    pparams = unmunchify(config.policy)

    if config.name == 'PPO' or config.name == 'NFVdeep':
        agent = baselines.MaskedPPO(env=env, seed=seed, **pparams)

    elif config['name'] == 'pre-trained':
        agent = baselines.MaskedPPO(env=env, seed=seed, **pparams)
        agent.load(config.model)

    elif config.name == 'FutureCoord':
        rpolicy = setup_agent(config.rollout, env, seed)
        agent = mcts.FutureCoord(rpolicy=rpolicy, seed=seed, **pparams)

    elif config.name == 'FutureMavenS':
        rpolicy = setup_agent(config.rollout, env, seed)
        agent = mcts.FutureMavenS(rpolicy=rpolicy, seed=seed, **pparams)

    elif config.name == 'Random':
        agent = baselines.RandomPolicy(seed=seed, **pparams)

    elif config.name == 'GreedyHeuristic':
        agent = baselines.GreedyHeuristic(seed=seed, **pparams)

    elif config.name == 'AllCombinations':
        agent = baselines.AllCombinations(**pparams)

    elif config.name == 'MavenS':
        agent = mcts.MavenS(seed=seed, **pparams)

    elif config.name == 'GRC_RS':
        agent = grc_rs.GRC(**pparams)
    elif config.name == 'GRC':
        agent = grc.GRC(**pparams)

    else:
        raise ValueError('Unknown agent.')

    return agent

def setup_agent_ac(config, env, seed):
    pparams = unmunchify(config.policy)
    if config.name == 'PPO':
        agent = baselines.MaskedPPO(env=env, seed=seed, **pparams)
        agent.name = 'PPO'
    elif config.name == 'NFVdeep':
        agent = baselines.MaskedPPO(env=env, seed=seed, **pparams)
        agent.name = 'NFVdeep'
    elif config['name'] == 'pre-trained':
        agent = baselines.MaskedPPO(env=env, seed=seed, **pparams)
        agent.load(config.model)
    elif config['name'] == 'DQN':
        agent = baselines.DQN(env=env, seed=seed, **pparams)
        agent.name = 'DQN'
    elif config.name == 'FutureCoord':
        rpolicy = setup_agent(config.rollout, env, seed)
        agent = mcts.FutureCoord(rpolicy=rpolicy, seed=seed, **pparams)

    elif config.name == 'FutureMavenS':
        rpolicy = setup_agent(config.rollout, env, seed)
        agent = mcts.FutureMavenS(rpolicy=rpolicy, seed=seed, **pparams)

    elif config.name == 'Random':
        agent = baselines.RandomPolicy(seed=seed, **pparams)

    elif config.name == 'GreedyHeuristic':
        agent = baselines.GreedyHeuristic(seed=seed, **pparams)

    elif config.name == 'AllCombinations':
        agent = baselines.AllCombinations(**pparams)
        agent.name = 'AllCombinations'

    elif config.name == 'MavenS':
        agent = mcts.MavenS(seed=seed, **pparams)

    elif config.name == 'GRC':
        agent = grc.GRC(**pparams)
    elif config.name == 'MaskedA2C':
        agent = baselines.MaskedA2C(env=env, seed=seed, **pparams)
        agent.name = 'A2C'
    elif config.name == 'MaskedARS':
        agent = baselines.MaskedARS(env=env, seed=seed, **pparams)
        agent.name = 'ARS'
    elif config.name == 'MaskedDQN':
        agent = baselines.MaskedDQN(env=env, seed=seed, **pparams)
        agent.name = 'DQN'
    elif config.name == 'FCFSPOLICY':
        agent = baselines.FCFSPolicy(env=env, seed=seed, **pparams)
        agent.name = 'FCFS'
    elif config.name == 'LONGESTPOLICY':
        agent = baselines.LongestPolicy(env=env, seed=seed, **pparams)
        agent.name = 'LONGESTPOLICY'
    elif config.name == 'GreedyPolicy':
        agent = baselines.GreedyPolicy(env=env, seed=seed, **pparams)
        agent.name = 'SBSEA'
    elif config.name == 'MaskablePPO':
        agent = baselines.MaskablePPO2(env=env, seed=seed, **pparams)
        agent.name = 'MaskablePPO'
    else:
        raise ValueError('Unknown agent.')
    return agent
def evaluate_episode(agent, monitor, process,arrival_rate):
    start = timer()
    ep_reward = 0.0
    obs = monitor.reset()
    
    while not monitor.env.done:
        action = agent.predict(observation=obs, env=monitor.env, process=process, deterministic=True)
        obs, reward, _, _ = monitor.step(action)
        ep_reward += reward

    end = timer()

    # get episode statistics from monitor
    ep_results = monitor.get_episode_results(agent,arrival_rate)
    ep_results['time'] = end - start

    return ep_results


def evaluate_episode_MaskablePPO(agent, monitor, process,arrival_rate):
    start = timer()
    ep_reward = 0.0
    obs = monitor.reset()

    while not monitor.env.done:
        action_mask=monitor.env.choose_valid_actions()
        action = agent.predict(observation=obs, action_masks=action_mask)

        print(monitor.env.all_waiting_queues)
        print(monitor.env.start_time)
        obs, reward, _, _ = monitor.step(action[0])
        ep_reward += reward

    end = timer()

    # get episode statistics from monitor
    ep_results = monitor.get_episode_results(agent,arrival_rate)
    ep_results['time'] = end - start

    return ep_results

def save_ep_results(data, path):
    data = pd.DataFrame.from_dict(data, orient='index')
    data = data.reset_index()
    data = data.rename({'index': 'episode'}, axis='columns')
    
    path = path / 'results.csv'
    
    # create results.csv or append records 
    if not os.path.isfile(str(path)):
        data.to_csv(path, header='column_names', index=False)
    else:
        data.to_csv(path, mode='a', header=False, index=False)

def setup_sim_process(rng, sim_rng, exp, args, eval_process, services, eday, sdays, load, rate, latency,arrival_rate):
    exp = deepcopy(exp)
    services = deepcopy(services)
    
    if args.oracle:
        # case: oracle traffic - set simulation process to real trace for oracle mode
        sim_process = TrafficStub(eval_process.sample())

    elif exp.traffic == 'erroneous':
        # case: sim. process has erroneous traffic pattern (other day) 
        eday = sample_excluding(rng, 0, NUM_DAYS, eday)
        sdays = [sample_excluding(rng, 0, NUM_DAYS, sday) for sday in sdays]
        sim_process = setup_process(sim_rng, exp, services, eday, sdays, load, rate, latency)

    elif exp.traffic == 'accurate':
        # (default) case: sim. process has correct traffic pattern (but not the exact trace)
        sim_process = setup_process(sim_rng, exp, services, eday, sdays, load, rate, latency,arrival_rate)

    else:
        raise ValueError('Invalid Traffic Option')

    return sim_process  

def setup_simulation(config, overlay, process, vnfs, services):
    if config.name == 'NFVdeep':
        return nfvdeep.NFVdeepCoordination(overlay, process, vnfs, services)

    return ServiceCoordination(overlay, process, vnfs, services)

def setup_simulation_ac_queue_reject(config, overlay, process, vnfs, services):
    if config.name == 'NFVdeep':
        return nfvdeep.NFVdeepCoordination(overlay, process, vnfs, services)
    env=ServiceCoordination2(overlay, process, vnfs, services)
    # env = make_vec_env(env, n_envs=1)
    env2=ActionMasker(env,mask_fn)
    return env2
def setup_simulation_ac_reject(config, overlay, process, vnfs, services):
    if config.name == 'NFVdeep':
        return nfvdeep.NFVdeepCoordination(overlay, process, vnfs, services)
    env=ServiceCoordination3(overlay, process, vnfs, services)
    # env = make_vec_env(env, n_envs=1)
    return env

def sample_excluding(rng, lower, upper, excluding):
    numbers = [n for n in range(lower, upper) if n != excluding]    
    return rng.choice(numbers)

