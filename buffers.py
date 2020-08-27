import gym
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import random
import torch.nn.functional as F
from heapq import heappush, heappop, heapify  



class ReplayBuffer():
    def __init__(self, device=torch.device('cpu') ,nitems=5, max_len=100000):
        self.deqs = [deque([]) for i in range(nitems)]
        self.length = 0
        self.max_len = max_len
        self.nitems = nitems
        self.device = device

    def __len__(self):
        return self.length

    def _remover(self):
        while self.length > self.max_len:
            for deq in self.deqs:
                deq.popleft()
            self.length -= 1

    def add(self, *items):
        for i, item in enumerate(items):
            item = item.view(-1, item.shape[-1])
            for it in item:
                self.deqs[i].append(it.unsqueeze(0))#.to(self.device))
        self.length += items[0].shape[0]*items[0].shape[1]
        self._remover()

        l1 = len(self.deqs[0])
        for i in range(5):
            if l1 != len(self.deqs[i]) :
                raise ValueError("neeee")

    def get(self, bsize):
        lidx = random.sample(range(0, self.length-1), bsize)
        toret = []
        for i in range(self.nitems):
            # print(i, lidx)
            self.deqs[i]
            l = [self.deqs[i][j] for j in lidx]
            l = torch.cat(l, dim=0).detach()
            toret.append(l)

        return toret


class PrioritizedReplayBuffer():
    def __init__(self, nitems=5, max_len=100000):
        self.minheap = MinHeap()
        self.length = 0
        self.max_len = max_len
        self.nitems = nitems

    def __len__(self):
        return self.length

    def _remover(self):
        while self.length > self.max_len:
            self.minheap.heap.pop()
            self.length -= 1

    def _add(self, items):
        self.minheap.insertKey(items)
        self.length += 1
        self._remover()

    def get(self, bsize):
        toret = list(self.minheap.extractMin())
        for i in range(bsize-1):
            toret2 = list(self.minheap.extractMin())
            # print(toret2, toret, i)
            for itor, tor in enumerate(toret[1:]):
                toret[itor+1] = torch.cat((tor, toret2[itor+1]), dim=0)
        self.length -= bsize
        return toret[1:]

    def add(self, *items, vals=None):
        items = list(items)

        for i in range(len(items)):
            items[i] = items[i].view(-1, items[i].shape[-1])
        if vals is None:
            vals = torch.randn(items[0].shape[0], 1)
        else:
            vals = vals.view(-1, vals.shape[-1])


        for i, val in enumerate(vals):
            l = [val.item()]
            for item in items:
                l.append(item[i].unsqueeze(0))
            try:
                self._add(tuple(l))
            except:
                print(l)
                assert 1==4
                self._add(tuple(l))


class MinHeap:       
    def __init__(self): 
        self.heap = []  
      
    def insertKey(self, k): 
        try:
            import copy
            nes = copy.deepcopy(k)
            heappush(self.heap, k)  
        except:
            print("kkkk1", nes)
      
              
    def extractMin(self): 
        
        try:
            import copy
            nes = copy.deepcopy(self.heap)
            return heappop(self.heap)  
        except:
            print("kkkkl1", nes)


