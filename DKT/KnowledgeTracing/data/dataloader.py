import os
import numpy as np
import sys
# GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(GRANDFA) 
sys.path.append('../')
import torch
import torch.utils.data as Data
from Constant import Constants as C
from data.readdata import DataReader
from data.DKTDataSet import DKTDataSet

def getTrainLoader(train_data_path,dict_c4):
    handle = DataReader(train_data_path ,C.MAX_STEP, C.NUM_OF_QUESTIONS)
    trainques, trainans = handle.getTrainData() # size: trainqus.reshape([-1, self.maxstep])
    dtrain = DKTDataSet(trainques, trainans,dict_c4) # here, dtrain is the data after one-hot encoding
    trainLoader = Data.DataLoader(dtrain, batch_size=C.BATCH_SIZE, shuffle=True)
    return trainLoader

def getTestLoader(test_data_path,dict_c4_test):
    handle = DataReader(test_data_path, C.MAX_STEP, C.NUM_OF_QUESTIONS)
    testques, testans = handle.getTestData()
    dtest = DKTDataSet(testques, testans,dict_c4_test)
    testLoader = Data.DataLoader(dtest, batch_size=C.BATCH_SIZE, shuffle=False)
    return testLoader

def getLoader(dataset):
    trainLoaders = []
    testLoaders = []
    if dataset == 'assist2009':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2009/builder_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2009/builder_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'assist2015':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2015/assist2015_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2015/assist2015_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'static2011':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/statics2011/statics2011_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/statics2011/statics2011_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'kddcup2010':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/kddcup2010/kddcup2010_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/kddcup2010/kddcup2010_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'assist2017':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2017/assist2017_train.txt')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/assist2017/assist2017_test.txt')
        testLoaders.append(testLoader)
    elif dataset == 'LON_course0':
        trainLoader = getTrainLoader(C.Dpath + '/LON_course/course0.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/LON_course/course0.csv')
        testLoaders.append(testLoader)
    elif dataset == 'LON_course4':
        cor_rate_train= np.load('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/cor_rate_c4.npy')
        cor_rate_test= np.load('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/cor_rate_c4_test.npy')
        trainLoader = getTrainLoader('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/course4_train.csv',cor_rate_train)
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/course4_test.csv',cor_rate_test)
        testLoaders.append(testLoader)
        
    elif dataset == 'LON_course27':
        trainLoader = getTrainLoader(C.Dpath + '/LON_course/course27.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/LON_course/course27_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'LON_course_total':
        trainLoader = getTrainLoader(C.Dpath + '/LON_course/total_course_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/LON_course/total_course_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'LON_course_combined':
        trainLoader = getTrainLoader(C.Dpath + '/LON_course/combined_course_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader(C.Dpath + '/LON_course/combined_course_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'LON_course_comb_df1':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/df1_new_train.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/df1_new_test.csv')
        testLoaders.append(testLoader)
        
    elif dataset == 'LON_sem1':
        trainLoader = getTrainLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/sem1_train_small.csv')
        trainLoaders.append(trainLoader)
        testLoader = getTestLoader('/cluster/home/yutang/Deep-Knowledge-Tracing/DKT/KTDataset/LON_course/sem1_test.csv')
        testLoaders.append(testLoader)
    elif dataset == 'synthetic':
        # trainLoader = getTrainLoader(C.Dpath + '/synthetic/synthetic_train_v0.txt')
        trainLoader = getTrainLoader('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/DeepKnowledgeTracing-DKT-Pytorch/DKT/KTDataset/synthetic/synthetic_train_v0.txt')
        trainLoaders.append(trainLoader)
        # testLoader = getTestLoader(C.Dpath + '/synthetic/synthetic_test_v0.txt')
        testLoader = getTestLoader('D:/ETHz/Internship/adaptive-e-learning-for-educational-recommendation-system/DeepKnowledgeTracing-DKT-Pytorch/DKT/KTDataset/synthetic/synthetic_test_v0.txt')
        testLoaders.append(testLoader)
    return trainLoaders, testLoaders