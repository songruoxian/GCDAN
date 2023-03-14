import logging
import torch
import dgl
import numpy as np
from tqdm import tqdm

from dataset import Dataset

from utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from evaluation import direct_node_classification_evaluation
from models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def pretrain(model, graph_target, feat_target, graph_source, feat_source, y_source, optimizer, max_epoch, device, scheduler, num_classes, lr_f, weight_decay_f,
             max_epoch_f, linear_prob, logger=None):
    logging.info("start training..")

    args = build_args()
    graph_target = graph_target.to(device)
    x_target = feat_target.to(device)
    graph_source = graph_source.to(device)
    x_source = feat_source.to(device)
    y_source = y_source.to(device)
    epoch_iter = tqdm(range(max_epoch))
    best_target_acc = 0.0
    best_target_pre = 0.0
    best_target_rec = 0.0
    best_target_f1 = 0.0
    best_target_auc = 0.0
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph_target, x_target, graph_source, x_source, y_source)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        final_acc, estp_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc = direct_node_classification_evaluation(model, graph_target, x_target, device)
        if estp_acc > best_target_acc:
            best_target_acc = estp_acc
            best_target_pre = estp_test_pre
            best_target_rec = estp_test_rec
            best_target_f1 = estp_test_f1
            best_target_auc = estp_test_auc
            torch.save(model.state_dict(), 'models/'+args.source+'_'+args.target+'_UDA_visual.pt')
        if (epoch + 1) % args.max_epoch == 0:
            print(best_target_acc,best_target_pre,best_target_rec,best_target_f1,best_target_auc)
    return model


def preprocess(graph):
    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph

def u_cat_e(edges):
  return {'m': torch.hstack([edges.src['feature'],edges.data['feature']])}

def mean_udf(nodes):
    return {'neigh_features': nodes.mailbox['m'].mean(1)}

def data_split(y,train_size):
    seeds = args.seeds
    for i, seed in enumerate(seeds):
        set_random_seed(seed)
    random_node_indices = np.random.permutation(y.shape[0])
    training_size = int(len(random_node_indices) * train_size)
    train_node_indices = random_node_indices[:training_size]
    test_node_indices = random_node_indices[:training_size]
    train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    train_masks[train_node_indices] = 1
    test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
    test_masks[test_node_indices] = 1
    return train_masks,test_masks

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    target = args.target
    source = args.source

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    dataset_target = Dataset("data_/{}".format(args.target), name=target)
    dataset_source = Dataset("data_/{}".format(args.source), name=source)

    target_data = dataset_target[0]
    t_src = target_data.edge_index[0]
    t_dst = target_data.edge_index[1]
    graph_target = dgl.graph((t_src, t_dst))
    graph_target = dgl.to_bidirected(graph_target)
    graph_target = graph_target.remove_self_loop().add_self_loop()
    graph_target.create_formats_()

    source_data = dataset_source[0]
    s_src = source_data.edge_index[0]
    s_dst = source_data.edge_index[1]
    graph_source = dgl.graph((s_src, s_dst))
    graph_source = dgl.to_bidirected(graph_source)
    graph_source = graph_source.remove_self_loop().add_self_loop()
    graph_source.create_formats_()

    '''target data split'''
    t_train_masks, t_test_masks = data_split(y=target_data.y,train_size=1.0)
    s_train_masks, s_test_masks = data_split(y=source_data.y,train_size=1.0)
    print('graph_target: ', graph_target,' graph_source: ',graph_source)

    print('target_data: ',target_data,' source_data: ',source_data)
    graph_target.ndata['feat'] = target_data.x
    graph_target.ndata['label'] = target_data.y
    graph_target.ndata['train_mask'] = t_train_masks
    graph_target.ndata['test_mask'] = t_test_masks
    graph_source.ndata['feat'] = source_data.x
    graph_source.ndata['label'] = source_data.y
    graph_source.ndata['train_mask'] = s_train_masks

    num_features = args.features
    num_classes = args.classes
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []
    estp_pre_list = []
    estp_rec_list = []
    estp_f1_list = []
    estp_auc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None
        model = build_model(args).to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x_target = graph_target.ndata["feat"]
        x_source = graph_source.ndata["feat"]
        y_source = graph_source.ndata["label"]
        if not load_model:
            model = pretrain(model, graph_target, x_target, graph_source, x_source, y_source, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
                             weight_decay_f, max_epoch_f, linear_prob, logger)
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load('models/' + args.type + '/' + str(args.drop_edge_rate) + '_' + str(
                args.lr) + '_' + "checkpoint.pt"))
        if save_model:
            logging.info("Saveing Model ...")
            torch.save(model.state_dict(), "checkpoint.pt")

        model = model.to(device)
        model.eval()

        final_acc, estp_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc = direct_node_classification_evaluation(model, graph_target, x_target, device)
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)
        estp_pre_list.append(estp_test_pre)
        estp_rec_list.append(estp_test_rec)
        estp_f1_list.append(estp_test_f1)
        estp_auc_list.append(estp_test_auc)

        if logger is not None:
            logger.finish()

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    estp_pre, estp_pre_std = np.mean(estp_pre_list), np.std(estp_pre_list)
    estp_rec, estp_rec_std = np.mean(estp_rec_list), np.std(estp_rec_list)
    estp_f1, estp_f1_std = np.mean(estp_f1_list), np.std(estp_f1_list)
    estp_auc, estp_auc_std = np.mean(estp_auc_list), np.std(estp_auc_list)
    print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")
    print(f"# early-stopping_pre: {estp_pre:.4f}±{estp_pre_std:.4f}")
    print(f"# early-stopping_rec: {estp_rec:.4f}±{estp_rec_std:.4f}")
    print(f"# early-stopping_f1: {estp_f1:.4f}±{estp_f1_std:.4f}")
    print(f"# early-stopping_auc: {estp_auc:.4f}±{estp_auc_std:.4f}")

if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)
