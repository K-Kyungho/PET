#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PET
from typing import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

import pickle
import gc
import copy
import random
import math


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values
   

    
class PET(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        
        self.dataset = self.conf["dataset"]
        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph
        
        self.beta_bi = self.conf["beta_bi"]
        self.beta_ui = self.conf["beta_ui"]
        # generate the graph without any dropouts for testing
        self.ui_main_view_graph_ori()
        self.ub_main_view_graph_ori()
        self.bi_main_view_graph_ori()

        # generate the graph with the augmentation for training
        self.ui_main_view_graph()
        self.ub_main_view_graph()
        self.bi_main_view_graph()

        self.ui_main_view_graph_aug2()
        self.ub_main_view_graph_aug2()
        self.bi_main_view_graph_aug2()

        self.ui_sub_view_graph()
        self.ui_sub_view_graph_ori()
        
        self.ub_sub_view_graph()
        self.ub_sub_view_graph_ori()
        
        self.bi_sub_view_graph()
        self.bi_sub_view_graph_ori()

        self.init_md_dropouts()
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temps"]
        self.c_temp_int = self.conf["c_temps_int"]
        self.c_bpr = self.conf['c_bpr']
        self.alpha = self.conf['alpha']
        
        
    def init_md_dropouts(self):
        self.ui_dropout = nn.Dropout(self.conf["q_ui"], True)
        self.ub_dropout = nn.Dropout(self.conf["q_ub"], True)
        self.bi_dropout = nn.Dropout(self.conf["q_bi"], True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
                
        self.L1 = nn.Linear(3, 3, bias = True)
        self.L2 = nn.Linear(3, 3, bias = True)
        self.L3 = nn.Linear(3, 1, bias = True)        
        
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.xavier_uniform_(self.L3.weight)
        
    def ui_sub_view_graph(self):
        ui_graph = self.ui_graph
        device = self.device

        modification_ratio = self.conf["q_ui"]
        graph = self.ui_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        ui_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        
        self.ui_sub_graph = to_tensor(ui_graph).to(device)
        
    def ui_sub_view_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device

        graph = self.ui_graph.tocoo()
        user_size = ui_graph.sum(axis=1) + 1e-8
        ui_graph = sp.diags(1/user_size.A.ravel()) @ ui_graph
        
        self.ui_sub_graph_ori = to_tensor(ui_graph).to(device)
    
    def ub_sub_view_graph(self):
        ub_graph = self.ub_graph
        device = self.device

        modification_ratio = self.conf["q_ub"]
        graph = self.ub_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        ub_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        user_size = ub_graph.sum(axis=1) + 1e-8
        ub_graph = sp.diags(1/user_size.A.ravel()) @ ub_graph
        
        self.ub_sub_graph = to_tensor(ub_graph).to(device)
        
    def ub_sub_view_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device

        user_size = ub_graph.sum(axis=1) + 1e-8
        ub_graph = sp.diags(1/user_size.A.ravel()) @ ub_graph
        
        self.ub_sub_graph_ori = to_tensor(ub_graph).to(device)
    
    def bi_sub_view_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        modification_ratio = self.conf["q_bi"]
        graph = self.bi_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8 #
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        
        self.bi_sub_graph = to_tensor(bi_graph).to(device)
   
        
    def bi_sub_view_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        
        self.bi_sub_graph_ori = to_tensor(bi_graph).to(device)
    
    # View Enhancement
    def ui_main_view_graph(self):
        ui_graph = self.ui_graph
        ub_graph = self.ub_graph
        user_size = ub_graph.sum(axis=1) + 1e-8
        ub_graph = sp.diags(1/user_size.A.ravel()) @ ub_graph
        
        bi_graph = self.bi_graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8 #
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        
        ubi_graph = ui_graph + self.beta_ui * (ub_graph * bi_graph)
        
        device = self.device
        modification_ratio = self.conf["q_ui"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ubi_graph.shape[0], ubi_graph.shape[0])), ubi_graph], [ubi_graph.T, sp.csr_matrix((ubi_graph.shape[1], ubi_graph.shape[1]))]])
                
        graph = item_level_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.ui_main_graph = to_tensor(laplace_transform(item_level_graph)).to(device)
    
    #View Enhancement
    def ui_main_view_graph_aug2(self):
        ui_graph = self.ui_graph
        
        ub_graph = self.ub_graph
        
        user_size = ub_graph.sum(axis=1) + 1e-8
        ub_graph = sp.diags(1/user_size.A.ravel()) @ ub_graph
        
        bi_graph = self.bi_graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8 #
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        
        ubi_graph = ui_graph + self.beta_ui * (ub_graph * bi_graph)
        
        device = self.device
        modification_ratio = self.conf["q_ui"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ubi_graph.shape[0], ubi_graph.shape[0])), ubi_graph], [ubi_graph.T, sp.csr_matrix((ubi_graph.shape[1], ubi_graph.shape[1]))]])
                
        graph = item_level_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.ui_main_graph_aug = to_tensor(laplace_transform(item_level_graph)).to(device)
        
    # View Enhancement
    def ui_main_view_graph_ori(self):
        ui_graph = self.ui_graph
        
        ub_graph = self.ub_graph
        user_size = ub_graph.sum(axis=1) + 1e-8
        ub_graph = sp.diags(1/user_size.A.ravel()) @ ub_graph
        
        bi_graph = self.bi_graph
        bundle_size = bi_graph.sum(axis=1) + 1e-8 #
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        
        ubi_graph = ui_graph + self.beta_ui * (ub_graph * bi_graph)
        
        device = self.device

        item_level_graph = sp.bmat([[sp.csr_matrix((ubi_graph.shape[0], ubi_graph.shape[0])), ubi_graph], [ubi_graph.T, sp.csr_matrix((ubi_graph.shape[1], ubi_graph.shape[1]))]])

        self.ui_main_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)

    def ub_main_view_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["q_ub"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        
        graph = bundle_level_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.ub_main_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)
    
    def ub_main_view_graph_aug2(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["q_ub"]
            
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
    
        graph = bundle_level_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
    
        self.ub_main_graph_aug = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def ub_main_view_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.ub_main_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    
    def bi_main_view_graph(self):
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["q_bi"]
        
        bundle_item_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])

        graph = bundle_item_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bundle_item_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        self.bi_main_graph = to_tensor(laplace_transform(bundle_item_graph)).to(device)
    
    def bi_main_view_graph_aug2(self):
        bi_graph = self.bi_graph
        device = self.device
        modification_ratio = self.conf["q_bi"]
        
        bundle_item_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])

        graph = bundle_item_graph.tocoo()
        values = np_edge_dropout(graph.data, modification_ratio)
        bundle_item_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()
        
        self.bi_main_graph_aug = to_tensor(laplace_transform(bundle_item_graph)).to(device)
    
        
    def bi_main_view_graph_ori(self):
        bi_graph = self.bi_graph
        
        device = self.device
        bundle_item_graph = sp.bmat([[sp.csr_matrix((bi_graph.shape[0], bi_graph.shape[0])), bi_graph], [bi_graph.T, sp.csr_matrix((bi_graph.shape[1], bi_graph.shape[1]))]])
        self.bi_main_graph_ori = to_tensor(laplace_transform(bundle_item_graph)).to(device)
        
        
    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            features = features / (i+2)
            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.sum(all_features, dim=1).squeeze(1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature
    
    
    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bi_sub_graph_ori, IL_items_feature)
                                 
        else:
            IL_bundles_feature = torch.matmul(self.bi_sub_graph, IL_items_feature)
                                 

        return IL_bundles_feature

    # View Enhancement
    def get_BI_user_rep(self, BI_items_feature, test):
        if test:
            new_ui = torch.matmul(self.ub_sub_graph_ori, self.bi_sub_graph_ori)
            new_ui = self.ui_sub_graph_ori + self.beta_bi * new_ui
            BI_users_feature = torch.matmul(new_ui, BI_items_feature)
                                 
        else:
            new_ui = torch.matmul(self.ub_sub_graph, self.bi_sub_graph)
            new_ui = self.ui_sub_graph + self.beta_bi * new_ui
            BI_users_feature = torch.matmul(new_ui, BI_items_feature)

        return BI_users_feature

    
    def propagate(self, test=False):
        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.ui_main_graph_ori, self.users_feature, self.items_feature, self.ui_dropout, test)
        
        else:
            IL_users_feature_aug1, IL_items_feature_aug1 = self.one_propagate(self.ui_main_graph, self.users_feature, self.items_feature, self.ui_dropout, test)
            IL_users_feature_aug, IL_items_feature_aug = self.one_propagate(self.ui_main_graph_aug, self.users_feature, self.items_feature, self.ui_dropout, test)
        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.ub_main_graph_ori, self.users_feature, self.bundles_feature, self.ub_dropout, test)
        else:
            BL_users_feature_aug1, BL_bundles_feature_aug1 = self.one_propagate(self.ub_main_graph, self.users_feature, self.bundles_feature, self.ub_dropout, test)
            BL_users_feature_aug, BL_bundles_feature_aug = self.one_propagate(self.ub_main_graph_aug, self.users_feature, self.bundles_feature, self.ub_dropout, test)

         # =========================== Bundle Item Propagation ================================
        if test:
            BI_bundles_feature, BI_items_feature = self.one_propagate(self.bi_main_graph_ori, self.bundles_feature, self.items_feature, self.bi_dropout, test)
        else:
            BI_bundles_feature_aug1, BI_items_feature_aug1 = self.one_propagate(self.bi_main_graph, self.bundles_feature, self.items_feature, self.bi_dropout, test)
            BI_bundles_feature_aug, BI_items_feature_aug = self.one_propagate(self.bi_main_graph_aug, self.bundles_feature, self.items_feature, self.bi_dropout, test)

        if test:
            users_feature = [IL_users_feature, BL_users_feature]
            items_feature = [IL_items_feature, BI_items_feature]
            bundles_feature = [BL_bundles_feature, BI_bundles_feature]
        else:
            users_feature = [IL_users_feature_aug1, IL_users_feature_aug, BL_users_feature_aug1, BL_users_feature_aug]
            items_feature = [IL_items_feature_aug1, IL_items_feature_aug, BI_items_feature_aug1, BI_items_feature_aug]
            bundles_feature = [BL_bundles_feature_aug1, BL_bundles_feature_aug, BI_bundles_feature_aug1, BI_bundles_feature_aug]
            
        return users_feature, items_feature, bundles_feature
        
    def cal_c_loss(self, pos, aug):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) 
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) 

        pos_score = torch.exp(pos_score / self.c_temp) 
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) 

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_c_loss_int(self, pos, aug):

        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) 
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) 
        pos_score = torch.exp(pos_score / self.c_temp_int)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp_int), axis=1)

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss
        
    def cal_loss(self, users_feature, items_feature, bundles_feature, user_parameter):
        
        IL_users_feature_aug1, IL_users_feature_aug, BL_users_feature_aug1, BL_users_feature_aug, BI_users_feature_aug1 = users_feature

        IL_items_feature_aug1, IL_items_feature_aug, BI_items_feature_aug1, BI_items_feature_aug = items_feature # !!!

        IL_bundles_feature_aug1, BL_bundles_feature_aug1, BL_bundles_feature_aug, BI_bundles_feature_aug1, BI_bundles_feature_aug = bundles_feature
        
        IL_pred = torch.sum(IL_users_feature_aug1 * IL_bundles_feature_aug1, 2)
        BL_pred = torch.sum(BL_users_feature_aug1 * BL_bundles_feature_aug1, 2)
        BI_pred = torch.sum(BI_users_feature_aug1 * BI_bundles_feature_aug1, 2)
        
        IL_user_parameter = user_parameter[:, 0] 
        BL_user_parameter = user_parameter[:, 1]
        BI_user_parameter = user_parameter[:, 2]
        
        pred_pos = torch.mul(IL_user_parameter, IL_pred[:, 0]) + torch.mul(BL_user_parameter, BL_pred[:, 0]) + torch.mul(BI_user_parameter, BI_pred[:, 0])
        pred_neg = self.alpha * (torch.mul(IL_user_parameter, IL_pred[:, 1]) + torch.mul(BL_user_parameter, BL_pred[:, 1]) + torch.mul(BI_user_parameter, BI_pred[:, 1]))
    
        pred = torch.stack([pred_pos, pred_neg], dim = 1)
        bpr_loss_main = cal_bpr_loss(pred)
        
        # BPR_aux
        pred_item = torch.sum(IL_users_feature_aug1*IL_items_feature_aug1, 2)        
        bpr_loss_item = cal_bpr_loss(pred_item)
        
        bpr_loss_aux = (cal_bpr_loss(IL_pred) + cal_bpr_loss(BL_pred) + cal_bpr_loss(BI_pred)) / 3 + self.c_bpr*bpr_loss_item
        
        #Inter CL
        u_cross_view_cl_1 = self.cal_c_loss(IL_users_feature_aug1, BL_users_feature_aug1)
        u_cross_view_cl_2 = self.cal_c_loss(BL_users_feature_aug1, BI_users_feature_aug1)
        
        b_cross_view_cl_1 = self.cal_c_loss(IL_bundles_feature_aug1, BL_bundles_feature_aug1)
        b_cross_view_cl_2 = self.cal_c_loss(BL_bundles_feature_aug1, BI_bundles_feature_aug1)

        c_losses = [u_cross_view_cl_1, u_cross_view_cl_2, b_cross_view_cl_1, b_cross_view_cl_2]# i cross 뺌
        
        c_loss = sum(c_losses) / len(c_losses)
        
        # Intra CL
        
        u_self_view_cl_1 = self.cal_c_loss_int(IL_users_feature_aug1, IL_users_feature_aug)
        u_self_view_cl_2 = self.cal_c_loss_int(BL_users_feature_aug1, BL_users_feature_aug)
        
        b_self_view_cl_1 = self.cal_c_loss_int(BL_bundles_feature_aug1, BL_bundles_feature_aug)
        b_self_view_cl_2 = self.cal_c_loss_int(BI_bundles_feature_aug1, BI_bundles_feature_aug)
        
        c_losses_int = [u_self_view_cl_1, u_self_view_cl_2, b_self_view_cl_1, b_self_view_cl_2]# i cross 뺌
        
        c_loss_intra = sum(c_losses_int) / len(c_losses_int)
        
        up_sum = torch.sum(user_parameter, dim = 0, keepdim = True)

        target = torch.zeros(1, 3, requires_grad = False).to(self.device)
        up_reg = F.mse_loss(up_sum, target, reduction = 'mean')
        
        
        return bpr_loss_main, bpr_loss_aux, c_loss, c_loss_intra, up_reg
    
    def forward(self, batch):
        #Augmentation for sub-view
        self.ui_sub_view_graph()
        self.ub_sub_view_graph()
        self.bi_sub_view_graph()
            
        users, bundles, items = batch
        
        users_feature, items_feature, bundles_feature = self.propagate()

        IL_users_feature_aug1, IL_users_feature_aug, BL_users_feature_aug1, BL_users_feature_aug = users_feature
        IL_items_feature_aug1, IL_items_feature_aug, BI_items_feature_aug1, BI_items_feature_aug = items_feature
        BL_bundles_feature_aug1, BL_bundles_feature_aug, BI_bundles_feature_aug1, BI_bundles_feature_aug = bundles_feature
        
        BI_users_feature_aug1 = self.get_BI_user_rep(BI_items_feature_aug1, test = False).squeeze(0)
        IL_bundles_feature_aug1 = self.get_IL_bundle_rep(IL_items_feature_aug1, test = False).squeeze(0)

        self.users_feature_prop = [IL_users_feature_aug1, IL_users_feature_aug, BL_users_feature_aug1, BL_users_feature_aug, BI_users_feature_aug1]
        self.bundles_feature_prop = [IL_bundles_feature_aug1, BL_bundles_feature_aug1, BL_bundles_feature_aug, BI_bundles_feature_aug1, BI_bundles_feature_aug] 
        self.items_feature_prop = [IL_items_feature_aug1, IL_items_feature_aug, BI_items_feature_aug1, BI_items_feature_aug]
        
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in self.users_feature_prop]
        items_embedding = [i[items] for i in self.items_feature_prop]
        bundles_embedding = [i[bundles] for i in self.bundles_feature_prop]

        ILIL = torch.sum((IL_users_feature_aug1 * IL_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)
        ILBL = torch.sum((IL_users_feature_aug1 * BL_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)
        ILBI = torch.sum((IL_users_feature_aug1 * BI_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)
        
        BLBL = torch.sum((BL_users_feature_aug1 * BL_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)
        BLBI = torch.sum((BL_users_feature_aug1 * BI_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)
        
        BIBI = torch.sum((BI_users_feature_aug1 * BI_users_feature_aug1 / math.sqrt(self.embedding_size)), dim = 1)

        self.IL_emb = torch.stack((ILIL, ILBL, ILBI), dim = 1) 
        self.BL_emb = torch.stack((ILBL, BLBL, BLBI), dim = 1) 
        self.BI_emb = torch.stack((ILBI, BLBI, BIBI), dim = 1) 
        
        IL_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.IL_emb))))) 
        BL_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.BL_emb))))) 
        BI_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.BI_emb))))) 
        
        param = torch.sigmoid(torch.cat((IL_param, BL_param, BI_param), dim = 1)) 
        param = param[users, :].squeeze(1)
        
        user_parameter = F.softmax(param , dim = 1) # 2048 3
        
        bpr_loss_main, bpr_loss_aux, c_loss, c_loss_int, up_reg = self.cal_loss(users_embedding, items_embedding, bundles_embedding, user_parameter)

        return bpr_loss_main, bpr_loss_aux, c_loss, c_loss_int, up_reg


    def evaluate(self, propagate_result, users):

        users_feature, items_feature, bundles_feature = propagate_result

        users_feature_IL, users_feature_BL = users_feature
        items_feature_IL, items_feature_BI = items_feature
        bundles_feature_BL, bundles_feature_BI = bundles_feature
        
        users_feature_BI = self.get_BI_user_rep(items_feature_BI, test = True).squeeze(0)
        bundles_feature_IL = self.get_IL_bundle_rep(items_feature_IL, test = True).squeeze(0)

        users_feature_IL = users_feature_IL[users]
        users_feature_BL = users_feature_BL[users]
        users_feature_BI = users_feature_BI[users]
    
        
        ILIL = torch.sum((users_feature_IL * users_feature_IL / math.sqrt(self.embedding_size)), dim = 1)
        ILBL = torch.sum((users_feature_IL * users_feature_BL / math.sqrt(self.embedding_size)), dim = 1)
        ILBI = torch.sum((users_feature_IL * users_feature_BI / math.sqrt(self.embedding_size)), dim = 1)
        
        BLBL = torch.sum((users_feature_BL * users_feature_BL / math.sqrt(self.embedding_size)), dim = 1)
        BLBI = torch.sum((users_feature_BL * users_feature_BI / math.sqrt(self.embedding_size)), dim = 1)
        
        BIBI = torch.sum((users_feature_BI * users_feature_BI / math.sqrt(self.embedding_size)), dim = 1)
            
        self.IL_emb = torch.stack((ILIL, ILBL, ILBI), dim = 1) 
        self.BL_emb = torch.stack((ILBL, BLBL, BLBI), dim = 1) 
        self.BI_emb = torch.stack((ILBI, BLBI, BIBI), dim = 1)  
            
        IL_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.IL_emb))))) 
        BL_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.BL_emb))))) 
        BI_param = self.L3(F.relu(self.L2(F.relu(self.L1(self.BI_emb)))))
            
        param = torch.sigmoid(torch.cat((IL_param, BL_param, BI_param), dim = 1))          
            
        user_parameter = F.softmax(param, dim = 1) # 2048 3
            
        IL_up = user_parameter[:, 0].unsqueeze(1)
        BL_up = user_parameter[:, 1].unsqueeze(1)
        BI_up = user_parameter[:, 2].unsqueeze(1)

        IL_pred = torch.mm(users_feature_IL, bundles_feature_IL.t()) 
        BL_pred = torch.mm(users_feature_BL, bundles_feature_BL.t()) 
        BI_pred = torch.mm(users_feature_BI, bundles_feature_BI.t()) 
                
        pred = torch.mul(IL_up, IL_pred) + torch.mul(BL_up, BL_pred) + torch.mul(BI_up, BI_pred)
            
        scores = pred
        
        return scores
