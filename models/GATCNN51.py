# -*- coding: utf-8 -*-
# from torchtools import tt
import torch.nn
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.gatbackbone import SliceEmbeddingImagenet

from models.layers import GraphAttentionLayer

from models.backbone_resnet import BasicBlock


# from layers1 import GraphAttentionLayer, UG_GraphAttentionLayer
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GATEmbedding(nn.Module):

    def __init__(self, mid_nodes, emb_size, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GATEmbedding, self).__init__()
        self.mid_nodes = mid_nodes
        self.emb_size = emb_size
        self.dropout = dropout
        self.slice_featuer = SliceEmbeddingImagenet(self.emb_size)
        # self.slice_featuer = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = GraphAttentionLayer(nhid * nheads, self.emb_size, dropout=dropout, alpha=alpha, concat=False)
        self.out_att2 = GraphAttentionLayer(self.emb_size, nclass, dropout=dropout, alpha=alpha, concat=False)

        # self.dim_attentions = [GraphAttentionLayer(self.emb_size, self.emb_size, dropout=dropout, alpha=alpha, concat=True) for _ in
        #                    range(nheads)]
        # for i, attention in enumerate(self.dim_attentions):
        #     self.add_module('dim_attention_{}'.format(i), attention)
        # self.out_att3 = GraphAttentionLayer(self.emb_size * nheads, self.emb_size, dropout=dropout, alpha=alpha, concat=False)
        # self.out_att4 = GraphAttentionLayer(self.emb_size, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.layer_last = nn.Sequential(
                                        # nn.Linear(in_features=self.emb_size,out_features=self.emb_size*2),
                                        nn.Linear(in_features=self.emb_size,out_features=self.emb_size*2),
                                        nn.ReLU(True),
                                        nn.Dropout(0.3),
                                        # nn.Linear(in_features=self.emb_size*2,out_features=self.emb_size),
                                        # nn.ReLU(True),
                                        )
        self.fc = nn.Linear(emb_size*2, 2)
        self.slice_fc = nn.Linear(emb_size, 2)

    def maxminscale(self, data):
        last_dim = data.size(-1)
        data_max = torch.max(data, dim=-1, keepdim=True).values.repeat(1, last_dim)
        data_min = torch.min(data, dim=-1, keepdim=True).values.repeat(1, last_dim)
        data = (data - data_min) / (data_max - data_min)
        return data

    def forward(self, input_img, input_struct=None):
        batch_size, c, h, w = input_img.size()
        slices = torch.zeros([batch_size, c+self.mid_nodes, self.emb_size], device=input_img.device)

        slices_features = [self.slice_featuer(value)[0] for value in input_img.chunk(input_img.size(1), dim=1)] # 可以加中间损失
        slices_features_stack = torch.stack(slices_features, dim=1)
        # slices_features_stack = self.slice_featuer(input_img)
        # slices_features_stack = slices_features_stack.view(slices_features_stack.shape[0], slices_features_stack.shape[1], -1)

        # for slice loss
        slices_hidden = slices_features_stack.mean(dim=1)
        slices_out = self.slice_fc(slices_hidden)

        slices[:, self.mid_nodes:, :] = slices_features_stack

        adj = self.get_adj(slices.size(1)).to(input_img.device)  # N X N
        adj = adj.unsqueeze(0).repeat(slices.size(0), 1, 1)  # B X N X N

        x = F.dropout(slices, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        # 2
        batch_attention = torch.stack([att(x, adj)[1] for att in self.attentions], dim=1).mean(dim=1)  # B X 3 X  N X N
        x = torch.cat([att(x, adj)[0] for att in self.attentions], dim=-1)  # B X N X nhid * nheads
        x = F.dropout(x, self.dropout, training=self.training)
        image_feature = F.elu(self.out_att1(x, adj)[0])  # B X N X emb_size
        image_feature = F.dropout(image_feature, self.dropout, training=self.training)
        # pre = F.elu(self.out_att2(image_feature, adj))  # B X N X nclass
        # pre = pre[0,:].unsqueeze(0)
        # pre = pre[:, 0, :].squeeze(1)  # B X nclass

        image_feature_second = image_feature[:, 0, :].squeeze(1)  # B X emb_size

        # image_feaure = F.normalize(imaget_feature, p=1, dim=-1)

        image_feature_second = self.layer_last(image_feature_second)
        # image_feature_second = self.layer_last(torch.cat([image_feature_second, input_struct], dim=1))

        predction = self.fc(image_feature_second)

        return slices_hidden, image_feature_second, predction, slices_out, batch_attention
        # return torch.cat([image_feature, input_struct], dim=1), self.fc(image_feature)

    # 构建表示一张图的邻接矩阵
    def get_adj(self, N):
        # N = 3 + C
        mid = ((N - self.mid_nodes) // 2 + self.mid_nodes) - 1
        width = ((N - self.mid_nodes) // self.mid_nodes) // 2
        adjMetrix = []

        for i in range(self.mid_nodes):
            adjMetrix.append([])
            for j in range(N):
                if i == (self.mid_nodes - 1):
                    if j > (self.mid_nodes - 1):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                else:
                    if j == (i + 1) or (j > mid - (i + 1) * width and j < mid + (i + 1) * width):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)

        for i in range(self.mid_nodes, N):
            adjMetrix.append([])
            for j in range(N):
                if i == self.mid_nodes:
                    if j == (self.mid_nodes + 1):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                elif i == N - 1:
                    if j == N - 2:
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                else:
                    if j == i - 1 or j == i + 1:
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)

        # adjMetrix = np.load('/media/data1/Models_ly/classification/UG-GAT/adjMetrix.npy')
        # adjMetrix = torch.from_numpy(adjMetrix)
        adjMetrix = sp.coo_matrix(adjMetrix)
        adjMetrix = normalize(adjMetrix + sp.eye(adjMetrix.shape[0]))
        adjMetrix = adjMetrix.todense()
        adjMetrix = torch.from_numpy(adjMetrix)
        adjMetrix = adjMetrix.float()
        # adjMetrix = adjMetrix
        return adjMetrix


class MultiAttention(nn.Module):

    def __init__(self, mid_nodes, emb_size, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(MultiAttention, self).__init__()
        self.mid_nodes = mid_nodes
        self.emb_size = emb_size
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = GraphAttentionLayer(nhid * nheads, self.emb_size, dropout=dropout, alpha=alpha, concat=False)
        self.out_att2 = GraphAttentionLayer(self.emb_size, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, slices, adj):

        x = F.dropout(slices, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)  # B X N X nhid * nheads
        x = F.dropout(x, self.dropout, training=self.training)
        image_feature = F.elu(self.out_att1(x, adj))  # B X N X emb_size
        image_feature = F.dropout(image_feature, self.dropout, training=self.training)
        pre = F.elu(self.out_att2(image_feature, adj))  # B X N X nclass
        # pre = pre[0,:].unsqueeze(0)
        pre = pre[:, 0, :].squeeze(1)  # B X nclass

        # image_feature_second = image_feature[:, 0, :].squeeze(1)  # B X emb_size

        return image_feature, pre

class Encoder(nn.Module):

    def __init__(self, n_layer, mid_nodes, emb_size, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(Encoder, self).__init__()
        self.mid_nodes = mid_nodes
        self.emb_size = emb_size
        self.dropout = dropout
        self.slice_featuer = SliceEmbeddingImagenet(self.emb_size)
        # self.slice_featuer = nn.Sequential(BasicBlock(32, 32), BasicBlock(32, 32))
        self.n_layer = n_layer

        self.multi_attentions = nn.ModuleList([MultiAttention(mid_nodes, emb_size, nfeat, nhid, nclass, dropout, alpha,
                                                            nheads) for _ in range(self.n_layer)])
        # 可注释掉了
        self.readout = nn.Sequential(nn.Linear(in_features=self.emb_size * self.n_layer, out_features=self.emb_size),
                                     nn.ReLU(True),
                                    )

        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=self.emb_size+7,out_features=self.emb_size*2),
        #     # nn.Linear(in_features=self.emb_size + 7, out_features=self.emb_size * 2),
        #     nn.ReLU(True),
        #     nn.Dropout(0.3),
        #     # nn.Linear(in_features=self.emb_size*2,out_features=self.emb_size),
        #     # nn.ReLU(True),
        # )

        # self.out = nn.Linear(emb_size, 2)
        self.out = nn.Linear(emb_size+7, 2)
        # self.out = nn.Linear(emb_size*2, 2)

        self.slice_fc = nn.Linear(emb_size, 2)

    def forward(self, input_img, input_struct=None):
        batch_size, c, h, w = input_img.size()
        slices = torch.zeros([batch_size, c + self.mid_nodes, self.emb_size], device=input_img.device)

        slices_features = [self.slice_featuer(value)[0] for value in input_img.chunk(input_img.size(1), dim=1)]
        slices_features_stack = torch.stack(slices_features, dim=1)
        # slices_features_stack = self.slice_featuer(input_img)
        # slices_features_stack = slices_features_stack.view(slices_features_stack.shape[0],
        #                                                    slices_features_stack.shape[1], -1)

        # for slice loss
        slices_hidden = slices_features_stack.mean(dim=1)
        slices_out = self.slice_fc(slices_hidden)

        slices[:, self.mid_nodes:, :] = slices_features_stack

        adj = self.get_adj(slices.size(1)).to(input_img.device)  # N X N
        adj = adj.unsqueeze(0).repeat(slices.size(0), 1, 1)  # B X N X N

        layers_outputs = torch.empty((self.n_layer, input_img.shape[0], self.emb_size)).to(input_img.device) # (n_layer, batch, emb_size)
        prediction_list = []
        for index, multi_attention in enumerate(self.multi_attentions):
            # input_img :   [batch_size, num_way*num_shot+1, 32, 32, 32]
            # enc_layers_outputs :   [n_layers, batch_size, num_way*num_shot+1, emb_size]
            slices, prediction = multi_attention(slices, adj)
            layers_outputs[index] = slices[:, 0, :].squeeze(1)
            prediction_list.append(prediction)
        # o1_l0 = layers_outputs[0, 2, :]
        # o1_l1 = layers_outputs[1, 2, :]
        # print(o1_l0)
        # print(o1_l1)
        '''
        batch = layers_outputs.shape[1]
        # enc_outputs = layers_outputs.reshape(-1, self.n_layer * self.emb_size)
        enc_outputs = layers_outputs.permute(1, 0, 2).contiguous().view(-1, self.n_layer * self.emb_size)
        # o2_l0 = enc_outputs[2, :self.emb_size]
        # o2_l1 = enc_outputs[2, self.emb_size:]
        # print(o2_l0)
        # print(o2_l1)
        enc_feas = self.readout(enc_outputs)
        '''
        # enc_feas = layers_outputs.mean(dim=0)
        enc_feas = layers_outputs[-1]

        # enc_feas = self.projection(torch.cat([enc_feas, input_struct], dim=1))

        enc_prediction = self.out(torch.cat([enc_feas, input_struct], dim=1))

        # enc_prediction = torch.sigmoid(enc_prediction)

        return enc_feas, enc_prediction, slices_out

    # 构建表示一张图的邻接矩阵
    def get_adj(self, N):
        # N = 3 + C
        mid = ((N - self.mid_nodes) // 2 + self.mid_nodes) - 1
        width = ((N - self.mid_nodes) // self.mid_nodes) // 2
        adjMetrix = []

        for i in range(self.mid_nodes):
            adjMetrix.append([])
            for j in range(N):
                if i == (self.mid_nodes - 1):
                    if j > (self.mid_nodes - 1):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                else:
                    if j == (i + 1) or (j > mid - (i + 1) * width and j < mid + (i + 1) * width):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)

                # if i == (self.mid_nodes - 1):
                #     if (j > mid - width and j < mid + width):
                #         adjMetrix[i].append(1)
                #     else:
                #         adjMetrix[i].append(0)
                # else:
                #     if j == (i + 1) or (j > mid - (self.mid_nodes - i) * width and j < mid + (self.mid_nodes - i) * width):
                #         adjMetrix[i].append(1)
                #     else:
                #         adjMetrix[i].append(0)

        for i in range(self.mid_nodes, N):
            adjMetrix.append([])
            for j in range(N):
                if i == self.mid_nodes:
                    if j == (self.mid_nodes + 1):
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                elif i == N - 1:
                    if j == N - 2:
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)
                else:
                    if j == i - 1 or j == i + 1:
                        adjMetrix[i].append(1)
                    else:
                        adjMetrix[i].append(0)

        # adjMetrix = np.load('/media/data1/Models_ly/classification/UG-GAT/adjMetrix.npy')
        # adjMetrix = torch.from_numpy(adjMetrix)
        adjMetrix = sp.coo_matrix(adjMetrix)
        adjMetrix = normalize(adjMetrix + sp.eye(adjMetrix.shape[0]))
        adjMetrix = adjMetrix.todense()
        adjMetrix = torch.from_numpy(adjMetrix)
        adjMetrix = adjMetrix.float()
        # adjMetrix = adjMetrix
        return adjMetrix