import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np
from torch_geometric.nn import  GCNConv
from torch.nn.parameter import Parameter
import math


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class  AOAENet(nn.Module):
    def __init__(self,dataset, args,**kwargs):
        kwargs.setdefault('aggr', 'mean')
        super(AOAENet, self).__init__()

        self.dropout =  args.dropout
        self.hidden = args.hidden
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, args.hidden)
        self.device =args.device
        self.GRU = nn.GRU(args.hidden, args.hidden)
        self.l = args.l
        self.attw = torch.nn.Linear(args.hidden * 3, 3)
        self.lin3 = nn.Linear(args.hidden*2 ,dataset.num_classes)

        self.att_0, self.att_1, self.att_2 = 0, 0, 0
        self.att_vec_0, self.att_vec_1, self.att_vec_2 = (
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
            Parameter(torch.FloatTensor(1 * args.hidden, 1).to(self.device)),
        )
        self.att_vec = Parameter(torch.FloatTensor(3, 3).to(self.device))
        self.reset_parameters()

    def reset_parameters(self):

        std_att = 1.0 / math.sqrt(self.att_vec_2.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.att_vec_1.data.uniform_(-std_att, std_att)
        self.att_vec_0.data.uniform_(-std_att, std_att)
        self.att_vec_2.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)


    def attention3(self, output_0, output_1, output_2):
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_0), self.att_vec_0),
                            torch.mm((output_1), self.att_vec_1),
                            torch.mm((output_2), self.att_vec_2),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]


    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        # device = x.device
        edge_index = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index)
        edge_index= sparse_mx_to_torch_sparse_tensor(edge_index)
        edge_index=edge_index.to(self.device)

        x = F.dropout(x, self.dropout, training=self.training)
        temp0 = self.lin1(x)  # 自身
        temp1 = torch.matmul(edge_index, temp0)
        temp2 = torch.matmul(edge_index, temp1)
        self.att_0, self.att_1, self.att_2 = self.attention3((temp0), (temp1), (temp2))
        # print(self.att_0.size())
        temp = self.att_0 * temp0 + self.att_1 * temp1 + self.att_2 * temp2

        temp3 = [temp0, temp]
        # print(temp3)
        temp3 = torch.stack(temp3, dim=1)
        # print(temp3)
        # print(temp3.size())#torch.Size([2708, 3, 64])
        temp3 = temp3.transpose(0, 1)
        temp3 = F.dropout(temp3, self.dropout, training=self.training)
        temp3, h_n0 = self.GRU(temp3)
        # print(h_n0)
        # print(h_n0.size())#torch.Size([1, 2708, 64])
        h_n0 = h_n0.view(-1, self.hidden)  # torch.Size([2708, 64])


        temp3 = [temp1, temp]
        temp3 = torch.stack(temp3, dim=1)
        temp3 = temp3.transpose(0, 1)
        temp3 = F.dropout(temp3, self.dropout, training=self.training)
        temp3, h_n1 = self.GRU(temp3)
        h_n1 = h_n1.view(-1, self.hidden)  # torch.Size([2708, 64])


        temp3 = [temp2, temp]
        temp3 = torch.stack(temp3, dim=1)
        temp3 = temp3.transpose(0, 1)
        temp3 = F.dropout(temp3, self.dropout, training=self.training)
        temp3, h_n2 = self.GRU(temp3)
        h_n2 = h_n2.view(-1, self.hidden)  # torch.Size([2708, 64])



        for i in range(self.l-1):
            self.att_0, self.att_1, self.att_2 = self.attention3( (h_n0), (h_n1), (h_n2) )
            #print(self.att_0.size())
            h_n = self.att_0 * h_n0 + self.att_1 * h_n1 + self.att_2 * h_n2

            temp3=[h_n0, h_n]
            #print(temp3)
            temp3=torch.stack(temp3, dim=1)
            #print(temp3)
            #print(temp3.size())#torch.Size([2708, 3, 64])
            temp3=temp3.transpose(0, 1)
            temp3 = F.dropout(temp3, self.dropout, training=self.training)
            temp3, h_n0 = self.GRU(temp3)
            #print(h_n0)
            #print(h_n0.size())#torch.Size([1, 2708, 64])
            h_n0 = h_n0.view(-1, self.hidden)  #torch.Size([2708, 64])


            temp3 = [h_n1, h_n]
            temp3 = torch.stack(temp3, dim=1)
            temp3 = temp3.transpose(0, 1)
            temp3 = F.dropout(temp3, self.dropout, training=self.training)
            temp3, h_n1 = self.GRU(temp3)
            h_n1 = h_n1.view(-1, self.hidden)  # torch.Size([2708, 64])


            temp3 = [h_n2, h_n]
            temp3 = torch.stack(temp3, dim=1)
            temp3 = temp3.transpose(0, 1)
            temp3 = F.dropout(temp3, self.dropout, training=self.training)
            temp3, h_n2 = self.GRU(temp3)
            h_n2 = h_n2.view(-1, self.hidden)  # torch.Size([2708, 64])




        self.att_0, self.att_1, self.att_2 = self.attention3((h_n0), (h_n1), (h_n2))
        # print(self.att_0.size())
        h_n = self.att_0 * h_n0 + self.att_1 * h_n1 + self.att_2 * h_n2

        h_n = torch.cat((temp0, h_n), dim=1)  # [V, 2*H]
        h_n = F.dropout(h_n, self.dropout, training=self.training)
        ans = self.lin3(h_n)
        #print(ans.size())
        return torch.log_softmax(ans, dim=1)
