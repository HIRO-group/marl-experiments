
import collections
import random
import numpy as np


# Based on ReplayBuffer class 
# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class Dataset():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, s_prime_lst, g1_list, g2_list = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, s_prime, g1, g2 = transition
            s_lst.append(s)
            a_lst.append(a)
            s_prime_lst.append(s_prime)
            g1_list.append(g1)
            g2_list.append(g2)

        return np.array(s_lst), np.array(a_lst), \
               np.array(s_prime_lst), np.array(g1_list), \
               np.array(g2_list)