import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch, to_dense_adj


def loss_function(r1,labels,nodeCount, lengs,refMat,mu,sig):
    #whole=to_dense_batch(labels, lengs, fill_value=0, max_num_nodes=refMat)

    #loss2 = torch.nn.SmoothL1Loss()
    #mae=torch.nn.L1Loss()
    lossMSE=torch.nn.MSELoss()
    emp=r1[0][:nodeCount[0]]
    for k in range(1,r1.shape[0]):
      emp=torch.cat((emp,r1[k][:nodeCount[k]]),dim=0)

    cost1=lossMSE(emp.cuda(),labels)
    KLD = -0.5 *(torch.mean(torch.sum(1 + sig - mu.pow(2) - sig.exp(), 1)))

    return cost1  + KLD 
