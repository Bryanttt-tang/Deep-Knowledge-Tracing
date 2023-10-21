import numpy as np
from data.DKTDataSet import DKTDataSet
import itertools
import tqdm

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        trainqus = np.array([])
        trainans = np.array([])
        with open(self.path, 'r') as train:
            for length, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading train data:    ', mininterval=2):
                length = int(length.strip().strip(',')) # total learning trajectory of individual student
                ques = np.array(ques.strip().strip(',').split(',')).astype(np.int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(np.int)
                mod = 0 if length%self.maxstep == 0 else (self.maxstep - length%self.maxstep)
                zero = np.zeros(mod) - 1 # padding 0 if the learning sequence is not long
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus = np.append(trainqus, ques).astype(np.int) # an array, with total lenght= 1056+504+...
                trainans = np.append(trainans, ans).astype(np.int)
                #print(trainqus)
                #print(trainqus.reshape([-1, self.maxstep]))
                #breakpoint()
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])


    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r') as test:
            for length, ques, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 3), desc='loading test data:    ', mininterval=2):
                length = int(length.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                mod = 0 if length % self.maxstep == 0 else (self.maxstep - length % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(np.int)
                testans = np.append(testans, ans).astype(np.int)
                #breakpoint()
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])