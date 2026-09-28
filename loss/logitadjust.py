import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):
    """
    Note: the original repo hardcodes torch.cuda.FloatTensor(cls_num_list),
    which only worked because m_list wasn't a registered buffer -- so
    calling .cuda() on the module never moved it, forcing CUDA at
    construction time. Fixed below via register_buffer, so
    LogitAdjust(cls_num_list).cuda() works correctly and this class is
    portable/testable on CPU too.
    """

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.register_buffer("m_list", m_list.view(1, -1))
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)
