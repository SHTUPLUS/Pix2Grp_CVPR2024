#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import numpy as np
import torch.nn as nn


def constant_init(module, val, bias=0):
    nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(module.weight,
                                 a=a,
                                 mode=mode,
                                 nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(module.weight,
                                a=a,
                                mode=mode,
                                nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(module,
                 a=1,
                 mode='fan_in',
                 nonlinearity='leaky_relu',
                 bias=bias,
                 distribution='uniform')


def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def c2_msra_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def init_weights(m: nn.Module, zero_init_final_gamma=False):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
            hasattr(m, "final_bn") and m.final_bn and zero_init_final_gamma
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()



def show_params_status(model, show_grad_only=False):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        # if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):

        st[p_name] = ([str(x) for x in p.size()],
                        np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        if show_grad_only:
            if p_req_grad:
                strings.append(
                    "{:<80s}: {:<16s}({:8d}) ({})".format(
                        p_name, "[{}]".format(",".join(size)), int(
                            prod), "grad" if p_req_grad else "    "
                    )
                )
        else:
            strings.append(
                "{:<80s}: {:<16s}({:8d}) ({})".format(
                    p_name, "[{}]".format(",".join(size)), int(
                        prod), "grad" if p_req_grad else "    "
                )
            )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )
