# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between each pair of query and prototype
    x: (n_query * n_classes, feature_size)
    y: (n_classes, feature_size)
    '''
    # Expand x and y to have the same shape for broadcasting
    x = x.unsqueeze(1)  # (n_query * n_classes, 1, feature_size)
    y = y.unsqueeze(0)  # (1, n_classes, feature_size)
    
    # Compute the squared Euclidean distance
    dists = torch.pow(x - y, 2).sum(dim=2)  # (n_query * n_classes, n_classes)
    return dists.sqrt()

def cosine_dist(x, y):
    '''
    Compute cosine distance between each pair of query and prototype.
    x: (n_query * n_classes, feature_size)
    y: (n_classes, feature_size)
    '''
    # Normalize x and y along the feature dimension to unit vectors
    x_norm = F.normalize(x, p=2, dim=1)  # (n_query * n_classes, feature_size)
    y_norm = F.normalize(y, p=2, dim=1)  # (n_classes, feature_size)
    
    # Compute cosine similarity (dot product of normalized vectors)
    cosine_similarity = torch.matmul(x_norm, y_norm.T)  # (n_query * n_classes, n_classes)
    
    # Convert cosine similarity to cosine distance: distance = 1 - similarity
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

def prototypical_loss(input, target, n_support, dist_type="euclidean"):
    '''
    In_supportpired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        '''
        select the n_support first elem in target_cpu that are equal to c
        input: integer 
        output: torch.tensor of size torch.Size([5])
        '''
        # s
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # get the classes labels
    classes = torch.unique(target_cpu)
    # get the number of classes
    n_classes = len(classes)

    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    # get the prototypes : torch.Size([6, 8192]) (because there are 6 classes)
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input_cpu[query_idxs]

    if dist_type=="euclidean":
        dists = euclidean_dist(query_samples, prototypes)
    elif dist_type=="cosine":
        dists = cosine_dist(query_samples, prototypes)


    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val