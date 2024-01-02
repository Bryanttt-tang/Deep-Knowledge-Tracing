import numpy as np
from torch.utils.data.dataset import Dataset
from Constant import Constants as C
import torch

class DKTDataSet(Dataset):
    def __init__(self, ques, ans, dict):
        self.ques = ques
        self.ans = ans
        self.dict = dict

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index] # a list of questions
        answers = self.ans[index]
        onehot = self.onehot(questions, answers)
        # added = self.add_feature(dadasdasd)
        return torch.FloatTensor(onehot.tolist())

    def onehot(self, questions, answers):
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS + 1])
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                result[i][questions[i]] = 1
            elif answers[i] == 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS] = 1
            # ques_cur=questions[i]
            result[i][-1]=self.dict[questions[i]]
        return result