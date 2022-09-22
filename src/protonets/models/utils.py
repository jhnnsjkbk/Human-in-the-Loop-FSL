import torch


def distance(x, y, cfg):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if cfg.MODEL.DISTANCE == 'euclidean':
        return torch.pow(x - y, 2).sum(2)
    elif cfg.MODEL.DISTANCE == 'cosine':
        # scaled cosine distance = (1 - cosine similarity) * alpha
        return (1. - torch.nn.CosineSimilarity(dim=2)(x, y)) * cfg.MODEL.ALPHA
