import sys
import os
import torch
# curPath = os.path.abspath(os.path.dirname(__file__))
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(GRANDFA)
# print(curPath)
# sys.path.append('curPath')
# sys.path.append('../')
from model.RNNModel import DKT
from model.LGR import LGR
# from model.SAKT.SAKTmodel import SAKTModel
from data.dataloader import getTrainLoader, getTestLoader, getLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval
import wandb
# control randomness
random_seed = 6
# random.seed(random_seed)
# np.random.seed(random_seed)
torch.manual_seed(random_seed)


if torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(random_seed)
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
# Set seed for NumPy if used within PyTorch (e.g., during data processing)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Initialize Weights and Biases with your API key and project name
wandb.init(
    project="DKT-trial 1",
    name="c4_lgr_added_step",

)

# print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

# model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT, C.EMBED).to(device)
model = LGR(C.INPUT, C.HIDDEN, C.OUTPUT).to(device)

optimizer_adam = optim.Adam(model.parameters(), lr=C.LR)
optimizer_adgd = optim.Adagrad(model.parameters(),lr=C.LR)
# optimizer_adgd.state = {key: value.to(device) for key, value in optimizer_adgd.state.items()}
loss_func = eval.lossFunc()

trainLoaders, testLoaders = getLoader(C.DATASET)
# learned embedding for each resource
best_auc=0
for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer, train_loss = eval.train(trainLoaders, model, optimizer_adgd, loss_func,device)
    # model = model.half()
    # torch.save(model, f"epoch_{epoch}.pt")
    train_auc, train_f1, train_recall, train_precision,train_loss=eval.test(trainLoaders, model,loss_func, device)
    val_auc, val_f1, val_recall, val_precision,val_loss=eval.test(testLoaders, model,loss_func, device)
    # if val_auc>best_auc:
    #     best_auc=val_auc
    #     torch.save(model.state_dict(), 'df1_deep.pth')
    wandb.log({"train_loss": train_loss}, step = epoch)
    wandb.log({"val_loss": val_loss}, step=epoch)
    wandb.log({"train_auc": train_auc}, step = epoch)
    wandb.log({"val_auc": val_auc}, step = epoch)
    
#     # wandb.log({"train_f1": train_f1}, step=epoch)
#     # wandb.log({"train_recall": train_recall}, step=epoch)
#     # wandb.log({"train_precision": train_precision}, step=epoch)
    
#     # wandb.log({"val_f1": val_f1}, step=epoch)
#     # wandb.log({"val_recall": val_recall}, step=epoch)
#     # wandb.log({"val_precision": val_precision}, step=epoch)
wandb.finish()
