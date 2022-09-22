def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
import sklearn.cluster as cluster
import sklearn.gaussian_process as gp
import torch
import scipy
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score

from src.protonets.models import register_model
from src.config.defaults import cfg
from .utils import distance


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class Protonet(nn.Module):
    def __init__(self, encoder, cfg):
        super(Protonet, self).__init__()

        self.encoder = encoder
        self.cfg = cfg

    def loss(self, sample):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if torch.cuda.is_available():
            target_inds = target_inds.cuda()

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]

        dists = distance(zq, z_proto, self.cfg)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def human_in_the_loop(self, sample, methods, return_features=False):
        xs = Variable(sample['xs'])  # support
        xq = Variable(sample['xq'])  # query
        # test images to validate simulation performance on a hold-out set
        x_test = Variable(sample['xt'])  # test

        n_class = xs.size(0)
        assert xq.size(0) == x_test.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)
        n_test = x_test.size(1)

        # ground truth for support and query set (y) and test set (y_test)
        y = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_support + n_query)
        y_test = torch.arange(0, n_class).view(n_class, 1).expand(n_class, n_test)

        if torch.cuda.is_available():
            y = y.cuda()
            y_test = y_test.cuda()

        # group images from support and query set per class and class-dim for inference
        x = torch.cat([xs, xq], 1).view(-1, *xs.size()[2:])
        x_test = x_test.view(-1, *x_test.size()[2:])
        if torch.cuda.is_available():
            x = x.cuda()
            x_test = x_test.cuda()

        z_dim = 512
        if cfg.MC_DROPOUT_ITERATIONS:
            # Monte Carlo Dropout
            for method in methods:
                assert 'cluster' not in method and 'oracle' not in method, \
                    'MC Dropout is not working with clusters or oracle'

            iterations = cfg.MC_DROPOUT_ITERATIONS
            enable_dropout(self.encoder)

            with torch.no_grad():
                z_mcdo = torch.cat([self.encoder.forward(x).view(1, -1, z_dim) for i in range(iterations)], dim=0)
                z_test_mcdo = torch.cat([self.encoder.forward(x_test).view(1, -1, z_dim) for i in range(iterations)],
                                        dim=0)

            z_mcdo = z_mcdo.view(iterations, n_class, -1, z_dim)

        else:
            # Normal inference without dropout
            assert 'BALD' not in methods and 'variation ratio' not in methods, \
                'Use MC Dropout for BALD and variation ratio'

            with torch.no_grad():
                z = self.encoder.forward(x)
                z_test = self.encoder.forward(x_test)

            # reshape z for prototype calculations
            z = z.view(n_class, -1, z_dim)

        results = {}

        if return_features:
            results['features'] = z

        # evaluates every method
        for method in methods:
            # initialize accuracy list and masks
            method_acc_history = []
            acc_history = []
            test_acc_history = []
            roc_auc_history = []
            f1_histroy = []
            selection_history = []

            support_mask = torch.cat([torch.ones([n_class, n_support, 1]), torch.zeros([n_class, n_query, 1])], dim=1)
            query_mask = torch.cat([torch.zeros([n_class, n_support]), torch.ones([n_class, n_query])], dim=1).bool()
            # for calculating AUROC
            init_query_mask = torch.cat([torch.zeros([n_class, n_support]), torch.ones([n_class, n_query])], dim=1). \
                bool().flatten()

            if torch.cuda.is_available():
                support_mask = support_mask.cuda()
                query_mask = query_mask.cuda()

            if method == 'class iteration' or method == 'random':
                # Indexes of query images (iterates over all classes)
                iteration_idx = torch.arange(n_class * (n_support + n_query))[query_mask.flatten()] \
                    .reshape(n_class, -1).T.flatten()
                random_idx = iteration_idx[torch.randperm(n=n_class * n_query)]

            # simulate iterative labeling
            for step in range((n_class * n_query) + 1):
                if cfg.MC_DROPOUT_ITERATIONS:
                    # calculate prototypes for each MC Dropout model
                    p_y_mcdo = []
                    test_p_y_mcdo = []
                    for i in range(iterations):
                        z_proto = (z_mcdo[i] * support_mask).sum(1) / support_mask.sum(1)

                        # get distance values for remaining query images
                        dists = distance(z_mcdo[i].view(-1, z_dim), z_proto, self.cfg)
                        p_y = F.softmax(-dists, dim=1)
                        p_y_mcdo.append(p_y.view(-1, n_class, 1))

                        # get distance for test images
                        test_dists = distance(z_test_mcdo[i].view(-1, z_dim), z_proto, self.cfg)
                        test_p_y = F.softmax(-test_dists, dim=1)
                        test_p_y_mcdo.append(test_p_y.view(-1, n_class, 1))

                    # combine predictions with mean
                    p_y_mcdo = torch.cat(p_y_mcdo, -1)
                    p_y = p_y_mcdo.mean(-1)
                    p_y_max, y_hat = p_y.max(1)

                    # combine mean for test images
                    test_p_y_mcdo = torch.cat(test_p_y_mcdo, -1)
                    test_p_y = test_p_y_mcdo.mean(-1)
                    test_p_y_max, test_y_hat = test_p_y.max(1)
                else:
                    # calculate prototypes
                    z_proto = (z * support_mask).sum(1) / support_mask.sum(1)

                    # get distance values for remaining query images
                    dists = distance(z.view(-1, z_dim), z_proto, self.cfg)
                    p_y = F.softmax(-dists, dim=1)

                    # get predictions
                    p_y_max, y_hat = p_y.max(1)

                    # get predictions for test images
                    test_dists = distance(z_test, z_proto, self.cfg)
                    test_p_y = F.softmax(-test_dists, dim=1)
                    test_p_y_max, test_y_hat = test_p_y.max(1)

                # Clustering
                if 'cluster' in method:
                    input_clustering = z.view(n_class * (n_support + n_query), -1).cpu().detach().numpy()
                    centroids = z_proto.cpu().detach().numpy()
                    kmeans = cluster.KMeans(n_clusters=cfg.DATASET.TEST_WAY, init=centroids)
                    clusters = kmeans.fit_predict(input_clustering)
                    distances = kmeans.fit_transform(input_clustering)
                    y_hat = torch.from_numpy(clusters)
                    distances = torch.from_numpy(distances)
                    if torch.cuda.is_available():
                        distances = distances.cuda()
                        y_hat = y_hat.cuda()
                    p_y = F.softmax(-distances, dim=1)

                    # Clustering for test images
                    input_clustering = z_test.view(n_class * n_test, -1).cpu().detach().numpy()
                    centroids = z_proto.cpu().detach().numpy()
                    kmeans = cluster.KMeans(n_clusters=cfg.DATASET.TEST_WAY, init=centroids)
                    clusters = kmeans.fit_predict(input_clustering)
                    test_distances = kmeans.fit_transform(input_clustering)
                    test_y_hat = torch.from_numpy(clusters)
                    if torch.cuda.is_available():
                        test_y_hat = test_y_hat.cuda()

                # Gaussian processes
                if 'gp' in method:
                    input_gp = z.view(n_class * (n_support + n_query), -1)[support_mask.flatten().bool()] \
                        .cpu().detach().numpy()
                    target_gp = y.flatten()[support_mask.flatten().bool()].cpu().detach().numpy()
                    z_gp = z.view(n_class * (n_support + n_query), -1).cpu().detach().numpy()
                    kernel = gp.kernels.RationalQuadratic()
                    gpc = gp.GaussianProcessClassifier(kernel=kernel, random_state=0).fit(input_gp, target_gp)
                    probabilities = gpc.predict_proba(z_gp)
                    probabilities = torch.from_numpy(probabilities)
                    if torch.cuda.is_available():
                        probabilities = probabilities.cuda()
                    probabilities_max, y_hat = probabilities.max(1)

                # calculate accuracy
                # already selected (and "labeled") images are counted as right classified in method accuracy (by adding
                # the support mask = labeled images), even though they could be false classified by the model prediction
                method_acc_history.append((torch.eq(y_hat.view(n_class, -1), y) + support_mask.view(n_class, -1))
                                          .bool()[:, n_support:].float().mean().item())

                acc_history.append(torch.eq(y_hat.view(n_class, -1), y)[:, n_support:].float().mean().item())

                test_acc_history.append(torch.eq(test_y_hat.view(n_class, -1), y_test).float().mean().item())

                one_hot_y_true = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=n_class).view(-1, n_class)
                roc_auc_history.append(roc_auc_score(y_true=one_hot_y_true[init_query_mask].cpu().detach().numpy(),
                                                     y_score=p_y[init_query_mask].cpu().detach().numpy()))

                f1_histroy.append(f1_score(y_true=y.flatten()[init_query_mask].cpu().detach().numpy(),
                                           y_pred=y_hat[init_query_mask].cpu().detach().numpy(),
                                           average='macro'))

                if query_mask.sum().item() == 0:
                    # exit last loop after labeling all images
                    break

                # Instance selection of one query image which is added to the support set
                if method == 'class iteration':
                    # select next image in query while iterating over all classes
                    selected_idx = iteration_idx[-query_mask.sum().item()]

                elif method == 'random':
                    # select random image from query
                    selected_idx = random_idx[-query_mask.sum().item()]

                elif method == 'oracle model':
                    def predict_oracle_model(idx):
                        # return 0 if image is already in support set
                        if idx in support_mask.flatten().nonzero():
                            return 0
                        # add image to support set
                        oracle_support_mask = support_mask.clone()
                        oracle_support_mask.flatten()[int(idx)] = 1.
                        # compute prototypes and predictions
                        oracle_z_proto = (z * oracle_support_mask).sum(1) / oracle_support_mask.sum(1)
                        oracle_dists = distance(z.view(-1, z_dim), oracle_z_proto, self.cfg)
                        _, oracle_y_hat = F.softmax(-oracle_dists, dim=1).max(1)
                        # return accuracy
                        return torch.eq(oracle_y_hat.view(n_class, -1), y)[:, n_support:].float().mean()

                    # get tensor with indexes for each image
                    oracle_pred = torch.arange(len(support_mask.flatten())).float()
                    if torch.cuda.is_available():
                        oracle_pred.cuda()
                    # get accuracy for each query image
                    oracle_pred.apply_(predict_oracle_model)
                    selected_idx = oracle_pred.argmax()

                elif method == 'oracle method':
                    def predict_oracle_method(idx):
                        # return 0 if image is already in support set
                        if idx in support_mask.flatten().nonzero():
                            return 0
                        # add image to support set
                        oracle_support_mask = support_mask.clone()
                        oracle_support_mask.flatten()[int(idx)] = 1.
                        # compute prototypes and predictions
                        oracle_z_proto = (z * oracle_support_mask).sum(1) / oracle_support_mask.sum(1)
                        oracle_dists = distance(z.view(-1, z_dim), oracle_z_proto, self.cfg)
                        _, oracle_y_hat = F.softmax(-oracle_dists, dim=1).max(1)
                        # return method accuracy
                        return (torch.eq(oracle_y_hat.view(n_class, -1), y) \
                                + oracle_support_mask.view(n_class, -1).bool())[:, n_support:].float().mean()

                    # get tensor with indexes for each image
                    oracle_pred = torch.arange(len(support_mask.flatten())).float()
                    if torch.cuda.is_available():
                        oracle_pred.cuda()
                    # get accuracy for each query image
                    oracle_pred.apply_(predict_oracle_method)
                    selected_idx = oracle_pred.argmax()

                elif method == 'maxentropy':
                    # select image from with highest entropy
                    entropy = torch.distributions.Categorical(probs=p_y).entropy()
                    selected_idx = (entropy * query_mask.flatten()).argmax()

                elif method == 'min confidence':
                    # select image with lowest confidence in prediction
                    selected_idx = (p_y_max + support_mask.flatten()).argmin()

                elif method == 'margin':
                    # Select image with the lowest margin between the most likely and second most likely class
                    p_y_sorted = p_y.sort(descending=True).values
                    selected_idx = ((p_y_sorted[:, 0] - p_y_sorted[:, 1]) + support_mask.flatten()).argmin()

                elif method == 'maxentropy distance':
                    # select image from with highest entropy
                    entropy = torch.distributions.Categorical(probs=dists).entropy()
                    selected_idx = (entropy * query_mask.flatten()).argmax()

                elif method == 'maxmin distance':
                    # select image furthest away from the closest prototype
                    selected_idx = (dists.min(1).values * query_mask.flatten()).argmax()

                elif method == 'margin distance':
                    # Select image with the lowest margin between the most likely and second most likely class
                    dists_sorted = dists.sort(descending=False).values
                    selected_idx = ((dists_sorted[:, 0] - dists_sorted[:, 1]) +
                                    support_mask.flatten() * dists_sorted.max()).argmin()

                elif method == 'variation ratio':
                    # Selects the sample with lack of confidence: proportion of predictions not in the mode class
                    class_counts = torch.stack([torch.bincount(x, minlength=n_class) for x in p_y_mcdo.max(1)[1]])
                    # (1 - #max_class_vote/#classes)
                    var_ratio = 1 - class_counts.max(-1).values.float() / iterations
                    selected_idx = (var_ratio - support_mask.flatten()).argmax()

                elif method == 'BALD':
                    entropy = torch.distributions.Categorical(probs=p_y).entropy()
                    expected_entropy = torch.distributions.Categorical(probs=p_y_mcdo).entropy().mean(-1)
                    selected_idx = ((entropy - expected_entropy)
                                    - support_mask.flatten() * (entropy.max() + expected_entropy.max())).argmax()

                elif method == 'cluster maxentropy':
                    # Select images with the max entropy between clusters.
                    entropy = torch.distributions.Categorical(probs=p_y).entropy()
                    selected_idx = (entropy * query_mask.flatten()).argmax()

                elif method == 'cluster margin':
                    # Select images with the lowest margin between the most likely and second most likely labels
                    p_y_sorted = p_y.sort(descending=True).values
                    selected_idx = ((p_y_sorted[:, 0] - p_y_sorted[:, 1]) + support_mask.flatten()).argmin()

                elif method == 'cluster maxdistance':
                    # similar to the maxmin distance method for cluster distances
                    one_hot_cluster = torch.nn.functional.one_hot(y_hat.to(torch.int64), num_classes=n_class)
                    cluster_distances, _ = (distances * one_hot_cluster).max(axis=1)
                    selected_idx = (cluster_distances * query_mask.flatten()).argmax()

                elif method == 'gp margin':
                    probabilities_sorted = probabilities.sort(descending=True).values
                    selected_idx = ((probabilities_sorted[:, 0] - probabilities_sorted[:, 1]) +
                                    support_mask.flatten()).argmin()

                elif method == 'gp maxentropy':
                    entropy = torch.distributions.Categorical(probs=probabilities).entropy()
                    selected_idx = (entropy * query_mask.flatten()).argmax()

                elif method == 'gp minmax':
                    # select image with lowest maximal probability
                    _, selected_idx = (probabilities_max + support_mask.flatten() * probabilities_max.max()).min(0)

                # move selected image from query to support set
                support_mask.view(-1)[selected_idx] = 1.
                query_mask.view(-1)[selected_idx] = False

                selection_history.append(selected_idx)

            results[method + '_Method_Accuracy'] = np.array(method_acc_history)
            results[method + '_Accuracy'] = np.array(acc_history)
            results[method + '_Test_Accuracy'] = np.array(test_acc_history)
            results[method + '_ROC_AUC'] = np.array(roc_auc_history)
            results[method + '_F1'] = np.array(f1_histroy)
            results[method + '_Selections'] = np.array(selection_history)

        return results


@register_model('protonet_conv')
def load_protonet_conv(cfg):
    x_dim = cfg.MODEL.INPUT_DIM
    hid_dim = cfg.MODEL.HIDDEN_DIM
    z_dim = cfg.MODEL.Z_DIM

    def conv_block(in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        if cfg.MODEL.DROPOUT:
            block.add_module(nn.Dropout(p=cfg.MODEL.DROPOUT))
        return block

    encoder = nn.Sequential(
        conv_block(x_dim[2], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    return Protonet(encoder, cfg)


class ResBlock(nn.Module):
    """
    Residual Block is similar to Oreshkin, B. N., Rodriguez, P., & Lacoste, A. (2018). Tadam: Task dependent adaptive
    metric for improved few-shot learning. (code https://github.com/ElementAI/TADAM/blob/master/model/tadam.py).
    It defers in the applied batch norm, where Oreshkin et al. are using a tasl-conditional batch norm.
    """

    def __init__(self, in_channels, out_channels, pooling='max', dropout=0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        if pooling == 'max':
            self.pool = nn.MaxPool2d(2)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(10)

    def forward(self, input):
        # block with depth 3 with 3x3 kernels
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        x = self.conv3(x)
        x = self.bn3(x)

        # shortcut connection
        x = x + self.shortcut(input)

        x = self.activation(x)
        x = self.dropout(x)
        x = self.pool(x)

        return x


@register_model('protonet_resnet')
def load_res_net(cfg):
    x_dim = cfg.MODEL.INPUT_DIM
    num_filters = [64, 128, 256, 512]

    encoder = nn.Sequential(
        ResBlock(x_dim[2], num_filters[0], dropout=cfg.MODEL.DROPOUT),
        ResBlock(num_filters[0], num_filters[1], dropout=cfg.MODEL.DROPOUT),
        ResBlock(num_filters[1], num_filters[2], dropout=cfg.MODEL.DROPOUT),
        # average pooling in last block to reduce embedding space to 512
        ResBlock(num_filters[2], num_filters[3], pooling='avg', dropout=cfg.MODEL.DROPOUT),
        Flatten()
    )

    return Protonet(encoder, cfg)
