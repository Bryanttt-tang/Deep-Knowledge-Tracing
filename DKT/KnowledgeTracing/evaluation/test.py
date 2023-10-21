import sys
import numpy as np
import pandas as pd
import torch
sys.path.append('../')
from model.RNNModel import DKT
from data.dataloader import getTrainLoader, getTestLoader, getLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval


model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)
checkpoint = torch.load('best.pth')

# Load the state dictionary into the model
model.load_state_dict(checkpoint)

trainLoaders, testLoaders = getLoader(C.DATASET)
val_recall=eval.test(testLoaders, model)
#for batch_idx, batch_data in enumerate(testLoaders[0]): # testLoaders is a list of size one
    # batch_idx is the batch number
    # batch_data is the batch of data
    # batch_targets is the batch of target values
    # print(f"Batch {batch_idx}:")
    # print("Batch Data:", batch_data)
    #torch.save(batch_data,'x_test.pth')
    #print("Batch Targets:", batch_targets)
    # import ipdb; ipdb.set_trace()

# x=torch.load('x_test.pth')
# # print(x)
# # print(x.shape)
# x_test=x[1,:,:]
# x_test=x_test.view(-1,50,100) # the model() takes a 3-d tensor as an input [n_students,max_step,2*n_questions]--one hot encoding
# print(x_test.shape)
# pred=model(x_test)
# print(pred)
# print(pred.size())

#
# # Convert the substrings to integers and create a 1-by-50 NumPy array
# ques_re = np.array([int(element) for element in ques_list])

# Now, one_by_50_array contains the extracted elements as integers in a 1-by-50 array
# print(ques_re)
# print(X_test.shape)
# print(ans)
# print(ques.shape)
# print(X_test.to_numpy()[0,:])
# ques_re=ques.reshape([-1, C.MAX_STEP])
# ans_re=ans.reshape([-1, C.MAX_STEP])


# def onehot(questions, answers):
#     result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS])
#     for i in range(C.MAX_STEP):
#         if answers[i] > 0:
#             result[i][questions[i]] = 1
#         elif answers[i] == 0:
#             result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1
#     return result
#
# x_test=onehot(ques_re,ans_re)
# print(x_test)
# print(x_test.shape)
# print(x_test.shape)
# print(x_test)