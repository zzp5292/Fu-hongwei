from typing import List, Dict
from functools import cmp_to_key

import numpy as np
import scipy.stats as stats
from numpy.random import default_rng, BitGenerator
from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson, SimuPoissonProcess


class Request:
    def __init__(self, arrival: float, duration: float, datarate: float, max_latency: float,
                 service: int):
        self.arrival = arrival
        self.duration = duration
        self.datarate = datarate
        self.max_latency = max_latency
        self.service: int = int(service)
        self.vtypes: List[int] = None
        self.resd_lat: float = self.max_latency

    def __str__(self):
        attrs = [round(self.duration, 2), round(self.datarate, 2), round(self.resd_lat, 2), round(self.max_latency, 2)]
        attrs = [*attrs, self.service]

        return 'Duration: {}; Rate: {}; Resd. Lat.: {}; Lat.: {}; Service: {}'.format(*attrs)


class ServiceTraffic:
    def __init__(self, rng: BitGenerator, service: int, horizon: float, process: Dict, datarates: Dict, latencies: Dict,
                 rates: np.ndarray, spaths: Dict,arrival_rate):
        self.rng = rng
        self.MAX_SEED = 2 ** 30 - 1

        self.service = service
        self.horizon = horizon
        self.process = process
        self.datarates = datarates
        self.latencies = latencies
        self.spaths = spaths
        self.arrival_rate=arrival_rate

        # create time function for inhomogenous poisson process
        T = np.linspace(0.0, horizon - 1, horizon)
        rates = np.ascontiguousarray(rates)
        self.rate_function = TimeFunction((T, rates))

    def sample_arrival2(self, horizon):
        poi_seed = self.rng.integers(0, self.MAX_SEED)
        poi_seed = int(poi_seed)

        in_poisson = SimuInhomogeneousPoisson(
            [self.rate_function], end_time=horizon, verbose=False, seed=poi_seed)
        in_poisson.track_intensity()
        in_poisson.simulate()
        arrivals = in_poisson.timestamps[0]
        return arrivals

    def sample_duration(self, size):
        lat = np.zeros(shape=[size])
        np.random.seed(10)
        for i in range(size):
            if self.service == 3:
                lat[i] = np.random.choice([25])
            elif self.service == 2:
                lat[i] = np.random.choice([20])
            elif self.service==0:
                lat[i] = np.random.choice([15])
            else:
                lat[i]=10
        lat = lat

        return lat

    def sample_datarates(self, size):
        lat = np.zeros(shape=[size])
        np.random.seed(10)
        for i in range(size):
            if self.service == 3:
                lat[i] = np.random.choice([200])
            elif self.service == 2:
                lat[i] = np.random.choice([150])
            elif self.service==1:
                lat[i]=np.random.choice([50])
            else:
                lat[i] = np.random.choice([100])
        lat = lat
        return lat

    def sample_latencies(self, size):
        lat = np.zeros(shape=size)
        np.random.seed(10)
        for i in range(size):
            if self.service == 3:
                lat[i]=2
            elif self.service == 2:
                lat[i] =2
            elif self.service==1:
                lat[i]=2
            else:
                lat[i] =2
        lat = lat
        return lat


    def sample_endpoints(self, arrivals):
        ingresses, egresses = [], []

        for arrival in arrivals:
            # get endpoint probability matrix for respective timestep
            timestep = int(np.floor(arrival))
            prob = self.endpoints[timestep]

            # sample ingress / egress from probability matrix
            flatten = prob.ravel()
            index = np.arange(flatten.size)
            ingress, egress = np.unravel_index(
                self.rng.choice(index, p=flatten), prob.shape)
            ingresses.append(ingress)
            egresses.append(egress)

        return ingresses, egresses

    def sample_arrival(self, horizon,arrival_rate):
        poi_seed = self.rng.integers(0, self.MAX_SEED)
        poi_seed = int(10)
        in_poisson = SimuPoissonProcess(arrival_rate*self.arrival_rate, end_time=45, verbose=False, seed=poi_seed)
        in_poisson.track_intensity()
        in_poisson.simulate()
        arrivals = in_poisson.timestamps[0]
        #返回齐次泊松过程流量请求的到达时间
        return arrivals

    def sample(self):
        # sample parameters for each service from distribution functions
        if self.service==3:
            arrival=self.sample_arrival(self.horizon,1)
        if self.service==2:
            arrival = self.sample_arrival(self.horizon,1)
        if self.service==1:
            arrival=self.sample_arrival(self.horizon,1)
        if self.service==0:
            arrival = self.sample_arrival(self.horizon,1)
        duration = self.sample_duration(len(arrival))
        # ingresses, egresses = self.sample_endpoints(arrival)

        # use arrival time to index the endpoint probability matrix and traffic matrix
        rates = self.sample_datarates(size=len(arrival))
        # propagation = np.asarray([self.spaths[ingr][egr] for ingr, egr in zip(ingresses, egresses)])
        latencies = self.sample_latencies(len(arrival))

        # build request objects and append them to the traffic trace
        requests = []
        for arr, dr, rate, lat, in zip(arrival, duration, rates, latencies):
            req = Request(arr, dr, rate, lat, self.service)
            requests.append(req)
        return requests


class Traffic:
    def __init__(self, processes):
        self.processes = processes

    def sample(self):
        # generate requests for each type of service from respective processes
        requests = [process.sample() for process in self.processes]
        requests = [req for srequests in requests for req in srequests]

        # sort to-be-simulated service requests according to their arrival time
        requests = sorted(requests, key=cmp_to_key(
            lambda r1, r2: r1.arrival - r2.arrival))
        return requests

    def __iter__(self):
        trace = self.sample()
        return iter(trace)


class TrafficStub:
    def __init__(self, trace):
        self.trace = trace

    def sample(self):
        return self.trace

    def __iter__(self):
        return iter(self.trace)
