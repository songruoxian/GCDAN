import copy

import numpy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import torch
import torch.nn as nn

from draw_node_mapping import args
from graphmae.utils import create_optimizer, accuracy

def direct_node_classification_evaluation(model, graph, x, device):
    model.eval()
    encoder = model.encoder
    dd = model.encoder_to_decoder
    cls_model = model.cls_model

    encoder.to(device)
    dd.to(device)
    cls_model.to(device)
    graph = graph.to(device)
    x = x.to(device)
    labels = graph.ndata["label"]

    with torch.no_grad():
        enc_rep, _ = encoder(graph, x, return_hidden=True)
        rep_t = dd(enc_rep)
        pred = cls_model(rep_t)
        y_true = labels.squeeze().long()
        preds = pred.max(1)[1].type_as(y_true)
        final_acc = accuracy(pred, labels)
        estp_test_acc = accuracy(pred, labels)
        estp_test_pre = precision_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_rec = recall_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_f1 = f1_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_auc = multi_class_auc(y_true.cpu().numpy(), preds.cpu().numpy())

    return final_acc, estp_test_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc


def multi_class_auc(y_true, preds):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_all = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
    y_pre_all = label_binarize(preds, classes=[0, 1, 2, 3, 4])
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_test_all[:, i], y_pre_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # print('roc_auc: ', roc_auc[i])
    return sum(roc_auc.values()) / 5

def linear_probing_for_transductive_node_classiifcation(model, graph, feat, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)

    train_mask = graph.ndata["train_mask"]
    val_mask = graph.ndata["val_mask"]
    test_mask = graph.ndata["test_mask"]
    labels = graph.ndata["label"]

    print('train_mask: ', torch.sum(train_mask), ' test_mask: ', torch.sum(test_mask))
    best_val_acc = 0
    best_test_acc = 0
    best_val_epoch = 0
    best_model = None
    print('labels: ', labels)
    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        print('train_mask: ', train_mask, 'out[train_mask]: ', out[train_mask].argmax(dim=1), ' labels[train_mask]: ',
              labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
            print('test_acc: ', test_acc)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)
        if test_acc >= best_test_acc:
            best_model_test = copy.deepcopy(model)
            torch.save(best_model_test.state_dict(),
                       'test_models/' + args.source + '_' + args.target + '_UDA_visual.pt')

        if not mute:
            epoch_iter.set_description(
                f"# Epoch: {epoch}, train_loss:{loss.item(): .4f}, val_loss:{val_loss.item(): .4f}, val_acc:{val_acc}, test_loss:{test_loss.item(): .4f}, test_acc:{test_acc: .4f}")

    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        y_true = labels[test_mask].squeeze().long()
        preds = pred[test_mask].max(1)[1].type_as(y_true)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
        estp_test_pre = precision_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_rec = recall_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_f1 = f1_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='macro')
        estp_test_auc = multi_class_auc(y_true.cpu().numpy(), preds.cpu().numpy())
    if mute:
        print(
            f"# IGNORE: --- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")
    else:
        print(
            f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f},early-stopping-TestPre: {estp_test_pre:.4f},early-stopping-TestRec: {estp_test_rec:.4f},early-stopping-TestF1: {estp_test_f1:.4f}, early-stopping-TestAUC: {estp_test_auc:.4f},Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    return test_acc, estp_test_acc, estp_test_pre, estp_test_rec, estp_test_f1, estp_test_auc

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits
