import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from CSP.pruned_layers import *

def summary(net):
    assert isinstance(net, nn.Module)
    print("Layer id\tType\t\tParameter\tNon-zero parameter\tSparsity(\%)")
    layer_id = 0
    num_total_params = 0
    num_total_nonzero_params = 0
    num_plinear_params = 0
    num_linear_params = 0
    skipNext=False
    for n, m in net.named_modules():
        if isinstance(m, PrunedLinear):
            weight = m.linear.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            print("%d\t\tPLinear\t\t%d\t\t%d\t\t\t%f" %(layer_id, num_parameters, num_nonzero_parameters, sparisty))
            num_plinear_params += num_parameters
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
            skipNext = True
        elif isinstance(m, PrunedConv):
            weight = m.conv.weight.data.cpu().numpy()
            weight = weight.flatten()
            num_parameters = weight.shape[0]
            num_nonzero_parameters = (weight != 0).sum()
            sparisty = 1 - num_nonzero_parameters / num_parameters
            layer_id += 1
            print("%d\t\tConvolutional\t%d\t\t%d\t\t\t%f" % (layer_id, num_parameters, num_nonzero_parameters, sparisty))
            num_total_params += num_parameters
            num_total_nonzero_params += num_nonzero_parameters
        elif isinstance(m, nn.Linear):
            if not skipNext:
                weight = m.weight.data.cpu().numpy()
                weight = weight.flatten()
                num_parameters = weight.shape[0]
                num_nonzero_parameters = (weight != 0).sum()
                sparisty = 1 - num_nonzero_parameters / num_parameters
                layer_id += 1
                print("%d\t\tLinear\t\t%d\t\t%d\t\t\t%f" %(layer_id, num_parameters, num_nonzero_parameters, sparisty))
                num_linear_params += num_parameters
                num_total_params += num_parameters
                num_total_nonzero_params += num_nonzero_parameters
            else:
                skipNext=False
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            layer_id += 1
            print("%d\t\tBatchNorm\tN/A\t\tN/A\t\t\tN/A" % (layer_id))
        elif isinstance(m, nn.ReLU):
            layer_id += 1
            print("%d\t\tReLU\t\tN/A\t\tN/A\t\t\tN/A" % (layer_id))

    print("PrunedLinear:totalLinear ratio = {}".format(float(num_plinear_params) / (num_plinear_params + num_linear_params)))
    print("Total nonzero parameters: %d" %num_total_nonzero_params)
    print("Total parameters: %d" %num_total_params)
    total_sparisty = 1. - (num_total_nonzero_params / num_total_params)
    print("Total sparsity: %f" %total_sparisty)
