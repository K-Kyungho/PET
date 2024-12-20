#PET
import os
import yaml
import json
import argparse
from tqdm import tqdm
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np


import torch
import torch.optim as optim
from utility import Datasets
from models.model import PET
import pickle
from itertools import product

def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    # Other hyperparameter configurations are in config.yaml
    parser.add_argument("-g", "--gpu", default="0", type=str, help="which gpu to use")
    parser.add_argument("-d", "--dataset", default="Youshu", type=str, help="which dataset to use, options: iFashion, NetEase, iFashion")
    parser.add_argument("-m", "--model", default="PET", type=str, help="which model to use, options: PET")
    parser.add_argument("-i", "--info", default="", type=str, help="any auxilary info that will be appended to the log file name")
    args = parser.parse_args()

    return args


def main():
    conf = yaml.safe_load(open("./config.yaml"))
    print("Loading configuration")

    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]

    assert paras["model"] in ["PET"], "Pls select models from: PET"

    if "_" in dataset_name:
        conf = conf[dataset_name.split("_")[0]]
    else:
        conf = conf[dataset_name]
        
    conf["dataset"] = dataset_name
    conf["model"] = paras["model"]
    dataset = Datasets(conf)

    conf["gpu"] = paras["gpu"]
    conf["info"] = paras["info"]

    conf["num_users"] = dataset.num_users
    conf["num_bundles"] = dataset.num_bundles
    conf["num_items"] = dataset.num_items

    os.environ['CUDA_VISIBLE_DEVICES'] = conf["gpu"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["device"] = device
    print(conf)
    best_performance_rec = float('-inf')
    best_performance_ndcg = float('-inf')
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}
    best_epoch = 0
    
    for lr, l2_reg, ui_dropout, ub_dropout, bi_dropout, embedding_size, num_layers, c_lambda, c_temp, c_lambda_int, c_temp_int, c_bpr, up_regs, c_aux, beta_bi, beta_ui, alpha in \
            product(conf['lrs'], conf['l2_regs'], conf['q_ui'], conf['q_ub'], conf['q_bi'], conf["embedding_sizes"], conf["num_layerss"], conf["c_lambdas"], conf["c_temps"], conf["c_lambdas_int"], conf["c_temps_int"], conf['c_bpr'], conf['up_regs'], conf['c_aux'], conf['beta_bi'], conf['beta_ui'], conf["alpha"]):
             
        log_path = "./log/%s/%s" %(conf["dataset"], conf["model"])
        run_path = "./runs/%s/%s" %(conf["dataset"], conf["model"])
        checkpoint_model_path = "./checkpoints/%s/%s/model" %(conf["dataset"], conf["model"])
        checkpoint_conf_path = "./checkpoints/%s/%s/conf" %(conf["dataset"], conf["model"])
        
        if not os.path.isdir(run_path):
            os.makedirs(run_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        if not os.path.isdir(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)
        if not os.path.isdir(checkpoint_conf_path):
            os.makedirs(checkpoint_conf_path)
        
        conf["lrs"] = lr
        conf["l2_reg"] = l2_reg
        conf["embedding_size"] =embedding_size
        conf["alpha"] = alpha
        
        conf["q_ui"] = ui_dropout
        conf["q_ub"] = ub_dropout
        conf["q_bi"] = bi_dropout
        conf["num_layers"] = num_layers
        
        conf["c_lambdas"] = c_lambda
        conf["c_temps"] = c_temp
        
        conf["c_lambdas_int"] = c_lambda_int
        conf["c_temps_int"] = c_temp_int
        conf['c_bpr'] = c_bpr
    
        conf["up_regs"] = up_regs
        conf["c_aux"] = c_aux
        conf["beta_bi"] = beta_bi
        conf["beta_ui"] = beta_ui
        
        settings =  []
        settings += ["Neg_%d" %(conf["neg_num"]), str(conf["batch_size_train"]), str(lr), str(l2_reg), str(embedding_size)]
        settings += [str(ui_dropout), str(ub_dropout), str(bi_dropout), str(num_layers)]
        settings += [str(c_lambda), str(c_temp), str(c_bpr), str(up_regs), str(beta_bi), str(beta_ui), str(alpha)]

        setting = "_".join(settings)
        log_path = log_path + "/" + setting
        run_path = run_path + "/" + setting
        checkpoint_model_path = checkpoint_model_path + "/" + setting
        checkpoint_conf_path = checkpoint_conf_path + "/" + setting
            
        run = SummaryWriter(run_path)

        # model initialization
        if conf['model'] == 'PET':
            model = PET(conf, dataset.graphs).to(device)
        else:
            raise ValueError("Unimplemented model %s" %(conf["model"]))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=conf["l2_reg"])
        batch_cnt = len(dataset.train_loader)
        test_interval_bs = int(batch_cnt * conf["test_interval"])

        for epoch in range(conf['epochs']):
            # For augmentation per epoch
            model.ui_main_view_graph()
            model.ub_main_view_graph()
            model.bi_main_view_graph()
                   
            model.ui_main_view_graph_aug2()
            model.ub_main_view_graph_aug2()
            model.bi_main_view_graph_aug2()

            epoch_anchor = epoch * batch_cnt
            model.train(True)
            ED_drop = False
            
            pbar = tqdm(enumerate(dataset.train_loader), total=len(dataset.train_loader))      
            
            for batch_i, batch in pbar:

                model.train(True)
                optimizer.zero_grad()
                batch = [x.to(device) for x in batch]
                batch_anchor = epoch_anchor + batch_i
                bpr_loss_main, bpr_loss_aux, c_loss, c_loss_int, up_reg = model(batch)
                loss = bpr_loss_main + conf['c_aux']*bpr_loss_aux + conf['c_lambdas']*c_loss + conf['c_lambdas_int']*c_loss_int + conf["up_regs"]*up_reg
                
                loss.backward()
                optimizer.step()
                
                loss_scalar = loss.detach()
                bpr_loss_main_scalar = bpr_loss_main.detach()
                bpr_loss_aux_scalar = bpr_loss_aux.detach()
                c_loss_scalar = c_loss.detach()
                c_loss_int_scalar = c_loss_int.detach()

                # loss output
                run.add_scalar("loss_bpr_main", bpr_loss_main_scalar, batch_anchor)
                run.add_scalar("loss_bpr_aux", bpr_loss_aux_scalar, batch_anchor)
                run.add_scalar("loss_c", c_loss_scalar, batch_anchor)
                run.add_scalar("loss_c_int", c_loss_int_scalar, batch_anchor)
                run.add_scalar("loss", loss_scalar, batch_anchor)

                pbar.set_description("epoch: %d, loss: %.4f, loss_bpr_main: %.4f, loss_bpr_aux: %.4f, c_loss: %.4f, c_loss_int: %.4f" %(epoch, loss_scalar, bpr_loss_main_scalar, bpr_loss_aux_scalar, c_loss_scalar, c_loss_int_scalar))
            
                if (batch_anchor+1) % test_interval_bs == 0:  
                    metrics = {}
                    metrics["val"] = test(model, dataset.val_loader, conf)
                    metrics["test"] = test(model, dataset.test_loader, conf)
                    # evaluation
                    best_metrics, best_perform, best_epoch = log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch)
                    topk_ = 20
                    current_performance_rec = metrics["test"]["recall"][topk_]
                    current_performance_ndcg = metrics['test']['ndcg'][topk_]
                    
                    if current_performance_rec > best_performance_rec and current_performance_ndcg > best_performance_ndcg:
                        best_performance_rec = current_performance_rec
                        best_performance_ndcg = current_performance_ndcg
                    
        print("Best Performance on Test Set:", best_performance_rec, best_performance_ndcg)
                
def init_best_metrics(conf):
    best_metrics = {}
    best_metrics["val"] = {}
    best_metrics["test"] = {}
    for key in best_metrics:
        best_metrics[key]["recall"] = {}
        best_metrics[key]["ndcg"] = {}
    for topk in conf['topk']:
        for key, res in best_metrics.items():
            for metric in res:
                best_metrics[key][metric][topk] = 0
    best_perform = {}
    best_perform["val"] = {}
    best_perform["test"] = {}

    return best_metrics, best_perform


def write_log(run, log_path, topk, step, metrics):
    curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    val_scores = metrics["val"]
    test_scores = metrics["test"]

    for m, val_score in val_scores.items():
        test_score = test_scores[m]
        run.add_scalar("%s_%d/Val" %(m, topk), val_score[topk], step)
        run.add_scalar("%s_%d/Test" %(m, topk), test_score[topk], step)

    val_str = "%s, Top_%d, Val:  recall: %f, ndcg: %f" %(curr_time, topk, val_scores["recall"][topk], val_scores["ndcg"][topk])
    test_str = "%s, Top_%d, Test: recall: %f, ndcg: %f" %(curr_time, topk, test_scores["recall"][topk], test_scores["ndcg"][topk])

    log = open(log_path, "a")
    log.write("%s\n" %(val_str))
    log.write("%s\n" %(test_str))
    log.close()

    print(val_str)
    print(test_str)


def log_metrics(conf, model, metrics, run, log_path, checkpoint_model_path, checkpoint_conf_path, epoch, batch_anchor, best_metrics, best_perform, best_epoch):
    for topk in conf["topk"]:
        write_log(run, log_path, topk, batch_anchor, metrics)

    log = open(log_path, "a")

    # You can change topk value
    topk_ = 20
    print("top%d as the final evaluation standard" %(topk_))
    if metrics["test"]["recall"][topk_] > best_metrics["test"]["recall"][topk_] and metrics["test"]["ndcg"][topk_] > best_metrics["test"]["ndcg"][topk_]:
        torch.save(model.state_dict(), checkpoint_model_path)
        dump_conf = dict(conf)
        del dump_conf["device"]
        json.dump(dump_conf, open(checkpoint_conf_path, "w"))
        best_epoch = epoch
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for topk in conf['topk']:
            for key, res in best_metrics.items():
                for metric in res:
                    best_metrics[key][metric][topk] = metrics[key][metric][topk]

            best_perform["test"][topk] = "%s, Best in epoch %d, TOP %d: REC_T=%.5f, NDCG_T=%.5f" %(curr_time, best_epoch, topk, best_metrics["test"]["recall"][topk], best_metrics["test"]["ndcg"][topk])
            best_perform["val"][topk] = "%s, Best in epoch %d, TOP %d: REC_V=%.5f, NDCG_V=%.5f" %(curr_time, best_epoch, topk, best_metrics["val"]["recall"][topk], best_metrics["val"]["ndcg"][topk])
            print(best_perform["val"][topk])
            print(best_perform["test"][topk])
            log.write(best_perform["val"][topk] + "\n")
            log.write(best_perform["test"][topk] + "\n")
    log.close()

    return best_metrics, best_perform, best_epoch

# test 
def test(model, dataloader, conf):
    tmp_metrics = {}
    for m in ["recall", "ndcg"]:
        tmp_metrics[m] = {}
        for topk in conf["topk"]:
            tmp_metrics[m][topk] = [0, 0]

    device = conf["device"]
    model.eval()
    rs = model.propagate(test=True)

    for users, ground_truth_u_b, train_mask_u_b in dataloader:
        ground_truth_u_b = ground_truth_u_b.to(device)
        pred_b = model.evaluate(rs, users.to(device))
        pred_b -= 1e8 * train_mask_u_b.to(device) 
        tmp_metrics = get_metrics(tmp_metrics, ground_truth_u_b, pred_b, conf["topk"])

    metrics = {}
    for m, topk_res in tmp_metrics.items():
        metrics[m] = {}
        for topk, res in topk_res.items():
            metrics[m][topk] = res[0] / res[1]

    return metrics

# Recall, NDCG
def get_metrics(metrics, grd, pred, topks):
    tmp = {"recall": {}, "ndcg": {}}
    for topk in topks:
        _, col_indice = torch.topk(pred, topk)
        row_indice = torch.zeros_like(col_indice) + torch.arange(pred.shape[0], device=pred.device, dtype=torch.long).view(-1, 1)
        is_hit = grd[row_indice.view(-1), col_indice.view(-1)].view(-1, topk)

        tmp["recall"][topk] = get_recall(pred, grd, is_hit, topk)
        tmp["ndcg"][topk] = get_ndcg(pred, grd, is_hit, topk)

    for m, topk_res in tmp.items():
        for topk, res in topk_res.items():
            for i, x in enumerate(res):
                metrics[m][topk][i] += x

    return metrics


def get_recall(pred, grd, is_hit, topk):
    epsilon = 1e-8
    hit_cnt = is_hit.sum(dim=1)
    num_pos = grd.sum(dim=1)

    # remove those test cases who don't have any positive items
    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = (hit_cnt/(num_pos+epsilon)).sum().item()

    return [nomina, denorm]


def get_ndcg(pred, grd, is_hit, topk):
    def DCG(hit, topk, device):
        hit = hit/torch.log2(torch.arange(2, topk+2, device=device, dtype=torch.float)).to(device)
        return hit.sum(-1)

    def IDCG(num_pos, topk, device):
        hit = torch.zeros(topk, dtype=torch.float).to(device)
        hit[:num_pos] = 1
        return DCG(hit, topk, device)

    device = grd.device
    IDCGs = torch.empty(1+topk, dtype=torch.float).to(device)
    IDCGs[0] = 1  # avoid 0/0
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk, device)

    num_pos = grd.sum(dim=1).clamp(0, topk).to(torch.long)
    dcg = DCG(is_hit, topk, device)

    idcg = IDCGs[num_pos]
    ndcg = dcg/idcg.to(device)

    denorm = pred.shape[0] - (num_pos == 0).sum().item()
    nomina = ndcg.sum().item()

    return [nomina, denorm]


if __name__ == "__main__":
    main()
