import numpy as np
#from sklearn.cluster import KMeans
from CSP.pruned_layers import *
import torch.nn as nn


def prune(net, method='cascade', q=5.0):
    # Before the training started, generate the mask
    assert isinstance(net, nn.Module)
    for n, m in net.named_modules():
        if isinstance(m, PrunedConv) or isinstance(m, PrunedLinear):
            if method == 'percentage':
                m.prune_by_percentage(q)
            elif method == 'std':
                m.prune_by_std(q)
            elif method == 'dil':
                m.prune_towards_dilation() ########## WORKING ON IT
            elif method == 'asym_dil':
                m.prune_towards_asym_dilation() ########## WORKING ON IT
            elif method == 'sintf':
                m.prune_structured_interfilter(q) ########## WORKING ON IT
            elif method == 'chunk':
                m.prune_chunk(q=q)
            elif method == 'cascade':
                m.prune_cascade_l1(q=q)
                #m.prune_filter_chunk(q=q) ##### not a good idea to do 2-stage prune naively
            elif method == 'SSL':
                m.prune_SSL(q)
            else:
                print("prune method {} not supported".format(method))
                exit()
