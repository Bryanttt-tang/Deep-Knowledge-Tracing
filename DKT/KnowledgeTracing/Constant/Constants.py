import sys
import os
# curPath = os.path.abspath(os.path.dirname(__file__))
# GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append(curPath) 
Dpath = '../../KTDataset'

datasets = {
    'assist2009' : 'assist2009',
    'assist2015' : 'assist2015',
    'assist2017' : 'assist2017',
    'static2011' : 'static2011',
    'kddcup2010' : 'kddcup2010',
    'LON_course0' : 'LON_course0',
    'LON_course4' : 'LON_course4',
    'LON_course27' : 'LON_course27',
    'LON_course_total' : 'LON_course_total',
    'LON_course_combined' : 'LON_course_combined',
    'LON_course_comb_df1' : 'LON_course_comb_df1',
    'LON_sem1' : 'LON_sem1',
    'synthetic' : 'synthetic'
}

# question number of each dataset
numbers = {
    'assist2009' : 124,  
    'assist2015' : 100,
    'assist2017' : 102,
    'static2011' : 1224, 
    'kddcup2010' : 661,
    'LON_course0' : 214,
    'LON_course4' : 161,
    'LON_course27' : 193,
    'LON_course_total' : 2796,
    'LON_course_combined' : 772,
    'LON_course_comb_df1' : 493,
    'LON_sem1' : 1029,
    'synthetic' : 50
}

DATASET = datasets['LON_course_comb_df1']
NUM_OF_QUESTIONS = numbers['LON_course_comb_df1']
MAX_STEP = 10 # the sequence length of RNN model
BATCH_SIZE = 64
LR = 0.01
EPOCH = 250
#input dimension
INPUT = NUM_OF_QUESTIONS * 2
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 200
# nums of hidden layers
LAYERS = 1
# output dimension
OUTPUT = NUM_OF_QUESTIONS
