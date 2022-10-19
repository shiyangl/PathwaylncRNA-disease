import time
import argparse

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import h5py
import matplotlib.pyplot as plt
from scipy import interp

from utils.pytorchtools import EarlyStopping
from utils.data import load_LNCD_data
from utils.tools import index_generator, parse_minibatch_LNCD
from model import MAGNN_lp

# Params
num_ntype = 3
dropout_rate = 0.001

lr = 0.001

weight_decay = 0.001
etypes_lists = [[[0, 1], [2, 5, 4, 3], [2, 3]],
                [[1, 0], [4, 5]]]
use_masks = [[True,False, False],
             [True,False]]
no_masks = [[False] * 3, [False] *2 ]
num_lncrna = 6583
num_disease = 1191
expected_metapaths = [
    [(0, 1, 0), (0, 2, 1, 2, 0), (0, 2, 0)],
    [(1, 0, 1), (1, 2, 1)]
]


def run_model_LNCD(feats_type, hidden_dim, num_heads, attn_vec_dim, rnn_type,
                     num_epochs, patience, batch_size, neighbor_samples, repeat, save_postfix):
    adjlists_ld, edge_metapath_indices_list_ld, _, type_mask, train_val_test_pos_lncrna_disease, train_val_test_neg_lncrna_disease = load_LNCD_data()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = []
    in_dims = []
    if feats_type == 0:
        for i in range(num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device))
    elif feats_type == 1:
        for i in range(num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(device))
    train_pos_lncrna_disease = train_val_test_pos_lncrna_disease['train_pos_lncrna_disease']
    val_pos_lncrna_disease = train_val_test_pos_lncrna_disease['val_pos_lncrna_disease']
    test_pos_lncrna_disease = train_val_test_pos_lncrna_disease['test_pos_lncrna_disease']
    train_neg_lncrna_disease = train_val_test_neg_lncrna_disease['train_neg_lncrna_disease']
    val_neg_lncrna_disease = train_val_test_neg_lncrna_disease['val_neg_lncrna_disease']
    test_neg_lncrna_disease = train_val_test_neg_lncrna_disease['test_neg_lncrna_disease']
    y_true_test = np.array([1] * len(test_pos_lncrna_disease) + [0] * len(test_neg_lncrna_disease))

    auc_list = []
    ap_list = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    y_real = []
    y_proba = []
    for _ in range(repeat):
        net = MAGNN_lp(
            [3, 2], 6, etypes_lists, in_dims, hidden_dim, hidden_dim, num_heads, attn_vec_dim, rnn_type, dropout_rate)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        # training loop
        net.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True, save_path='checkpoint/checkpoint_metapath1_{}.pt'.format(save_postfix))
        dur1 = []
        dur2 = []
        dur3 = []
        train_pos_idx_generator = index_generator(batch_size=batch_size, num_data=len(train_pos_lncrna_disease))
        val_idx_generator = index_generator(batch_size=batch_size, num_data=len(val_pos_lncrna_disease), shuffle=False)
        for epoch in range(num_epochs):
            t_start = time.time()
            # training
            net.train()
            for iteration in range(train_pos_idx_generator.num_iterations()):
                # forward
                t0 = time.time()

                train_pos_idx_batch = train_pos_idx_generator.next()
                train_pos_idx_batch.sort()
                train_pos_lncrna_disease_batch = train_pos_lncrna_disease[train_pos_idx_batch].tolist()
                train_neg_idx_batch = np.random.choice(len(train_neg_lncrna_disease), len(train_pos_idx_batch))
                train_neg_idx_batch.sort()
                train_neg_lncrna_disease_batch = train_neg_lncrna_disease[train_neg_idx_batch].tolist()

                train_pos_g_lists, train_pos_indices_lists, train_pos_idx_batch_mapped_lists = parse_minibatch_LNCD(
                    adjlists_ld, edge_metapath_indices_list_ld, train_pos_lncrna_disease_batch, device, neighbor_samples, use_masks, num_lncrna)
                train_neg_g_lists, train_neg_indices_lists, train_neg_idx_batch_mapped_lists = parse_minibatch_LNCD(
                    adjlists_ld, edge_metapath_indices_list_ld, train_neg_lncrna_disease_batch, device, neighbor_samples, no_masks, num_lncrna)

                t1 = time.time()
                dur1.append(t1 - t0)

                [pos_embedding_lncrna, pos_embedding_disease], _ = net(
                    (train_pos_g_lists, features_list, type_mask, train_pos_indices_lists, train_pos_idx_batch_mapped_lists))
                [neg_embedding_lncrna, neg_embedding_disease], _ = net(
                    (train_neg_g_lists, features_list, type_mask, train_neg_indices_lists, train_neg_idx_batch_mapped_lists))
                pos_embedding_lncrna = pos_embedding_lncrna.view(-1, 1, pos_embedding_lncrna.shape[1])
                pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                neg_embedding_lncrna = neg_embedding_lncrna.view(-1, 1, neg_embedding_lncrna.shape[1])
                neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)
                pos_out = torch.bmm(pos_embedding_lncrna, pos_embedding_disease)
                neg_out = -torch.bmm(neg_embedding_lncrna, neg_embedding_disease)
                train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

                t2 = time.time()
                dur2.append(t2 - t1)

                # autograd
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                t3 = time.time()
                dur3.append(t3 - t2)

                # print training info
                if iteration % 100 == 0:
                    print(
                        'Epoch {:05d} | Iteration {:05d} | Train_Loss {:.4f} | Time1(s) {:.4f} | Time2(s) {:.4f} | Time3(s) {:.4f}'.format(
                            epoch, iteration, train_loss.item(), np.mean(dur1), np.mean(dur2), np.mean(dur3)))
            # validation
            net.eval()
            val_loss = []
            with torch.no_grad():
                for iteration in range(val_idx_generator.num_iterations()):
                    # forward
                    val_idx_batch = val_idx_generator.next()
                    val_pos_lncrna_disease_batch = val_pos_lncrna_disease[val_idx_batch].tolist()
                    val_neg_lncrna_disease_batch = val_neg_lncrna_disease[val_idx_batch].tolist()
                    val_pos_g_lists, val_pos_indices_lists, val_pos_idx_batch_mapped_lists = parse_minibatch_LNCD(
                        adjlists_ld, edge_metapath_indices_list_ld, val_pos_lncrna_disease_batch, device, neighbor_samples, no_masks, num_lncrna)
                    val_neg_g_lists, val_neg_indices_lists, val_neg_idx_batch_mapped_lists = parse_minibatch_LNCD(
                        adjlists_ld, edge_metapath_indices_list_ld, val_neg_lncrna_disease_batch, device, neighbor_samples, no_masks, num_lncrna)

                    [pos_embedding_lncrna, pos_embedding_disease], _ = net(
                        (val_pos_g_lists, features_list, type_mask, val_pos_indices_lists, val_pos_idx_batch_mapped_lists))
                    [neg_embedding_lncrna, neg_embedding_disease], _ = net(
                        (val_neg_g_lists, features_list, type_mask, val_neg_indices_lists, val_neg_idx_batch_mapped_lists))
                    pos_embedding_lncrna = pos_embedding_lncrna.view(-1, 1, pos_embedding_lncrna.shape[1])
                    pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                    neg_embedding_lncrna = neg_embedding_lncrna.view(-1, 1, neg_embedding_lncrna.shape[1])
                    neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)

                    pos_out = torch.bmm(pos_embedding_lncrna, pos_embedding_disease)
                    neg_out = -torch.bmm(neg_embedding_lncrna, neg_embedding_disease)
                    val_loss.append(-torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out)))
                val_loss = torch.mean(torch.tensor(val_loss))
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        test_idx_generator = index_generator(batch_size=batch_size, num_data=len(test_pos_lncrna_disease), shuffle=False)
        net.load_state_dict(torch.load('checkpoint/checkpoint_metapath1_{}.pt'.format(save_postfix)))
        net.eval()
        pos_proba_list = []
        neg_proba_list = []
        with torch.no_grad():
            for iteration in range(test_idx_generator.num_iterations()):
                # forward
                test_idx_batch = test_idx_generator.next()
                test_pos_lncrna_disease_batch = test_pos_lncrna_disease[test_idx_batch].tolist()
                test_neg_lncrna_disease_batch = test_neg_lncrna_disease[test_idx_batch].tolist()
                test_pos_g_lists, test_pos_indices_lists, test_pos_idx_batch_mapped_lists = parse_minibatch_LNCD(
                    adjlists_ld, edge_metapath_indices_list_ld, test_pos_lncrna_disease_batch, device, neighbor_samples, no_masks, num_lncrna)
                test_neg_g_lists, test_neg_indices_lists, test_neg_idx_batch_mapped_lists = parse_minibatch_LNCD(
                    adjlists_ld, edge_metapath_indices_list_ld, test_neg_lncrna_disease_batch, device, neighbor_samples, no_masks, num_lncrna)

                [pos_embedding_lncrna, pos_embedding_disease], _ = net(
                    (test_pos_g_lists, features_list, type_mask, test_pos_indices_lists, test_pos_idx_batch_mapped_lists))
                [neg_embedding_lncrna, neg_embedding_disease], _ = net(
                    (test_neg_g_lists, features_list, type_mask, test_neg_indices_lists, test_neg_idx_batch_mapped_lists))
                pos_embedding_lncrna = pos_embedding_lncrna.view(-1, 1, pos_embedding_lncrna.shape[1])
                pos_embedding_disease = pos_embedding_disease.view(-1, pos_embedding_disease.shape[1], 1)
                neg_embedding_lncrna = neg_embedding_lncrna.view(-1, 1, neg_embedding_lncrna.shape[1])
                neg_embedding_disease = neg_embedding_disease.view(-1, neg_embedding_disease.shape[1], 1)

                pos_out = torch.bmm(pos_embedding_lncrna, pos_embedding_disease).flatten()
                neg_out = torch.bmm(neg_embedding_lncrna, neg_embedding_disease).flatten()
                pos_proba_list.append(torch.sigmoid(pos_out))
                neg_proba_list.append(torch.sigmoid(neg_out))
            y_proba_test = torch.cat(pos_proba_list + neg_proba_list)
            y_proba_test = y_proba_test.cpu().numpy()
        auc = roc_auc_score(y_true_test, y_proba_test)
        ap = average_precision_score(y_true_test, y_proba_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_true_test, y_proba_test)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC MAGNN %d(auc=%0.4f)' % (i, roc_auc))
        print('Link Prediction Test')
        print('AUC = {}'.format(auc))
        print('AP = {}'.format(ap))

        auc_list.append(auc)
        ap_list.append(ap)
        precision, recall, _ = precision_recall_curve(y_true_test, y_proba_test)
        y_real.append(y_true_test)
        y_proba.append(y_proba_test)

    print('----------------------------------------------------------------')
    print('LNCD Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AP_mean = {}, AP_std = {}'.format(np.mean(ap_list), np.std(ap_list)))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (auc=%0.4f)' % mean_auc, lw=2, alpha=.8)
    with h5py.File('./results/pathwayLDA_mean_AUC.h5', 'w') as hf:
        hf['fpr'] = mean_fpr
        hf['tpr'] = mean_tpr
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    with h5py.File('./results/patpwayLDA_mean_PR.h5', 'w') as hf:
        hf['precision'] = precision
        hf['recall'] = recall



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MAGNN testing for the recommendation dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - all id vectors; ' +
                         '1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
    ap.add_argument('--epoch', type=int, default=100, help='Number of epochs. Default is 3.')
    ap.add_argument('--patience', type=int, default=5, help='Patience. Default is 5.')
    ap.add_argument('--batch-size', type=int, default=128, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=10, help='Repeat the training and testing for N times. Default is 10.')
    ap.add_argument('--save-postfix', default='LNCD', help='Postfix for the saved model and result. Default is LNCD.')

    args = ap.parse_args()
    run_model_LNCD(args.feats_type, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type, args.epoch,
                     args.patience, args.batch_size, args.samples, args.repeat, args.save_postfix)
