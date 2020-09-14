from diff_encoder import Encoder, GNN
from torch_geometric.utils import to_dense_batch, to_dense_adj
import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool

class GraphAE(torch.nn.Module):
    def __init__(self,):
        super(GraphAE, self).__init__()   
        #max_nodes = 2448
        #num_nodes = ceil(0.5 * max_nodes)
        

        self.gnn1_pool = GNN(3, 64, 612, add_loop=True)
        self.gnn1_embed = GNN(3, 64, 64, add_loop=True, lin=False)
        
        self.gnn2_embed = GNN(2 * 64, 64, 64, lin=False)
        self.gnn3_embed = GNN(2 * 64, 64, 64, lin=False)


        self.lin1=torch.nn.Linear(128,64)
        self.lin2=torch.nn.Linear(64,3)

    def upsample(self,X,A,S):
      
      Xout=torch.bmm(S,X)
      Aout=torch.bmm(S,torch.bmm(S,A).permute(0,2,1))
      return Xout,Aout

    def encode(self,x,adj,lengs,refMat,maxNodes):  
        self.enc=Encoder(maxNodes).cuda()
        whole=to_dense_batch(x, lengs, fill_value=0, max_num_nodes=refMat.shape[1])
        wholeAdj=to_dense_adj(adj, lengs, edge_attr=None, max_num_nodes=refMat.shape[1])

        r1,r2,adjMat,_,_= self.enc(whole[0].cuda(),wholeAdj.cuda(),refMat.cuda())
                
        return r1,r2,adjMat

    def reparametrize(self, mu, logvar):
      if self.training:
          return mu + torch.randn_like(logvar) * torch.exp(logvar)
      else:
          return mu  
    
    def decode(self,z,adj):
      
        out1=self.gnn3_embed(z,adj)

        

        out2=self.gnn2_embed(out1,adj)
        s=torch.nn.Parameter(torch.randn(out2.shape[0],2*out2.shape[1],out2.shape[1])).cuda()

        xout2,aout2=self.upsample(out2,adj,s)

        s2=torch.nn.Parameter(torch.randn(xout2.shape[0],2*xout2.shape[1],xout2.shape[1])).cuda()

        out3=self.gnn2_embed(xout2,aout2)
        out3,aout3=self.upsample(xout2,aout2,s2)

        out3=self.lin1(out3)
        return self.lin2(out3),aout3
        
    
    def forward(self,x,adj,lengs,refMat,maxNodes):
        mu,logvar,adjMat = self.encode(x,adj,lengs,refMat,maxNodes)     ## mu, log sigma 
        z = self.reparametrize(mu, logvar) ## z = mu + eps*sigma 
        z2,adj2=self.decode(z,adjMat)
        return z2, adj2, mu, logvar 
