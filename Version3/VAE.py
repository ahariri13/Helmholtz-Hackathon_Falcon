#from diff_encoder import Encoder, GNN
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os.path as osp
from math import ceil
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

import torch_geometric
import torch_geometric.nn as tnn

from torch_geometric.nn import EdgeConv, NNConv, GraphConv, DenseGCNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.nn import TopKPooling, GCNConv,GatedGraphConv, SAGPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat

class GraphAE(torch.nn.Module):
    def __init__(self,in_channels, out_channels1, out_channels2,out_channels3,out_channels4, out_channels5,out_channels6, dropout):
        super(GraphAE, self).__init__()   
        maxNodes = 2000

        self.out_channels2=out_channels2
        """
        Encoding
        """
        ### Encoding
        self.sage1=tnn.DenseGCNConv(in_channels,out_channels1)
        self.sage2=tnn.DenseGCNConv(out_channels1,out_channels2)
        
        self.poolit1=tnn.DenseGCNConv(out_channels2,ceil(maxNodes/4))
        self.poolit2=tnn.DenseGCNConv(out_channels4,ceil(maxNodes/8))
        self.poolit3=tnn.DenseGCNConv(out_channels5,10)
        
        self.sage3=tnn.DenseGCNConv(out_channels2,out_channels3)
        self.sage4=tnn.DenseGCNConv(out_channels3,out_channels4)

        self.sage5=tnn.DenseGCNConv(out_channels4,out_channels5)

        self.tr1=nn.Linear(out_channels5,out_channels6)
        self.tr2=nn.Linear(out_channels6,2048)

        self.rev1=nn.Linear(out_channels6,out_channels5)
        self.rev2=nn.Linear(2048,out_channels6)

        self.revsage1=tnn.DenseGCNConv(out_channels1,in_channels)
        self.revsage2=tnn.DenseGCNConv(out_channels2,out_channels1)

        self.revsage3=tnn.DenseGCNConv(out_channels3,out_channels2)
        self.revsage4=tnn.DenseGCNConv(out_channels4,out_channels3)

        self.revsage5=tnn.DenseGCNConv(out_channels5,out_channels4)

        self.drop=torch.nn.Dropout(p=0.2)

        ## Batch Normalization
        self.bano1 = nn.BatchNorm1d(num_features=2000)
        self.bano2 = nn.BatchNorm1d(num_features=2000)
        self.bano3 = nn.BatchNorm1d(num_features=500)
        self.bano4 = nn.BatchNorm1d(num_features=500)
        self.bano5 = nn.BatchNorm1d(num_features=250)
        self.bano6 = nn.BatchNorm1d(num_features=40)


    def upsample(self,X,A,S):
      
      Xout=torch.bmm(S,X)
      Aout=torch.bmm(S,torch.bmm(S,A).permute(0,2,1))
      return Xout,Aout

    def encode(self,x,adj,lengs,refMat,maxNodes):  
        whole=to_dense_batch(x, lengs, fill_value=0, max_num_nodes=2000)#refMat.shape[1])
        wholeAdj=to_dense_adj(adj, lengs, edge_attr=None, max_num_nodes=2000)#refMat.shape[1]).cuda()
        
        ### 1 
        hidden1=self.sage1(whole[0].cuda(),wholeAdj)
        hidden1=self.bano1(hidden1)
        hidden1=F.tanh(hidden1) ## BxNxL1
        hidden1=self.drop(hidden1)
      
        ### 2
        hidden2=self.sage2(hidden1,wholeAdj)
        hidden2=self.bano2(hidden2)
        hidden2=F.leaky_relu(hidden2) ## BxNxL2
        hidden2=self.drop(hidden2)

        ### Pool1
        pool1=self.poolit1(hidden2,wholeAdj)
        pool1=F.leaky_relu(pool1) ## BxNxC1
 
        out1,adj1,_,_=dense_diff_pool(hidden2,wholeAdj,pool1,mask=refMat)
           
        ### 3
        hidden3=self.sage3(out1,adj1)
        hidden3=self.bano3(hidden3)
        hidden3=F.leaky_relu(hidden3)
        hidden3=self.drop(hidden3)

        ### 4 
        hidden4=self.sage4(hidden3,adj1)
        hidden4=self.bano4(hidden4)
        hidden4=F.leaky_relu(hidden4) 
        hidden4=self.drop(hidden4)

        ### Pool2
        pool2=self.poolit2(hidden4,adj1)
        pool2=F.leaky_relu(pool2) ## BxN/4xC2

        out2,adj2,_,_=dense_diff_pool(hidden4,adj1,pool2)

        out2=self.sage5(out2,adj2)
        out2=self.bano5(out2)
        out2=F.leaky_relu(out2) 
        out2=self.drop(out2)

        ### Pool3
        pool3=self.poolit3(out2,adj2)
        pool3=F.leaky_relu(pool3) ## BxN/8xC3

        out3,adj3,_,_=dense_diff_pool(out2,adj2,pool3)

        ### 5
        hidden5=self.tr1(out2)
        hidden5=F.leaky_relu(hidden5) 
                
 
        return self.tr2(hidden5),self.tr2(hidden5),adj2#,info1,info2, info3,edge_index

    def reparametrize(self, mu, logvar):
      if self.training:
          return mu + torch.randn_like(logvar) * torch.exp(logvar)
      else:
          return mu  
    
    def decode(self,z,adj,maxNodes):

        out1=self.rev2(z)  
        out1=F.leaky_relu(out1)   
        out1=self.rev1(out1)
        out1=F.leaky_relu(out1)


        s0=torch.nn.Parameter(torch.randn(out1.shape[0],250,out1.shape[1])).cuda()
        
        out1,aout0=self.upsample(out1,adj,s0)
        out1=F.leaky_relu(out1)

        out1=self.revsage5(out1,aout0)
        out1=self.bano5(out1)
        out1=F.leaky_relu(out1)

        s=torch.nn.Parameter(torch.randn(out1.shape[0],500,out1.shape[1])).cuda()

        xout2,aout2=self.upsample(out1,aout0,s)
        xout2=F.leaky_relu(xout2)

        out2=self.revsage4(xout2,aout2)
        out2=self.bano4(out2)
        out2=F.leaky_relu(out2)

        out2=self.revsage3(out2,aout2)
        out2=self.bano4(out2)
        out2=F.leaky_relu(out2)

        s2=torch.nn.Parameter(torch.randn(out2.shape[0],4*out2.shape[1],out2.shape[1])).cuda()

        out3,aout3=self.upsample(out2,aout2,s2)
        out3=F.leaky_relu(out3)

        out3=self.revsage2(out3,aout3)
        out3=self.bano1(out3)
        out3=F.leaky_relu(out3)

        out3=self.revsage1(out3,aout3)
        out3=self.bano1(out3)
        out3=F.relu(out3)

        return out3,aout3

    def forward(self,x,adj,lengs,refMat,maxNodes):
        self.maxNodes=maxNodes
        mu,logvar,adjMat = self.encode(x,adj,lengs,refMat,maxNodes)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        z2,adj2=self.decode(z,adjMat,maxNodes)
        return z2, adj2, mu, logvar 
