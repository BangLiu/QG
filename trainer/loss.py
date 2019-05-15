# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def NMTLoss(vocabSize):
    """
    Use NLLLoss as loss function of generator.
    Set weight of PAD as zero.
    """
    PAD_idx = 0
    weight = torch.ones(vocabSize)
    weight[PAD_idx] = 0
    crit = nn.NLLLoss(weight, reduction="sum")
    if torch.cuda.is_available():
        crit.cuda()
    return crit


def QGLoss(g_prob_t, g_targets,
           c_outputs, c_switch,
           c_gate_values, c_targets,
           crit, copyCrit):
    # loss func with copy mechanism
    c_output_prob = c_outputs * c_gate_values.expand_as(c_outputs) + 1e-8
    g_output_prob = g_prob_t * (1 - c_gate_values).expand_as(g_prob_t) + 1e-8

    c_output_prob_log = torch.log(c_output_prob)
    g_output_prob_log = torch.log(g_output_prob)

    c_output_prob_log = c_output_prob_log * \
        (c_switch.unsqueeze(2).expand_as(c_output_prob_log))
    g_output_prob_log = g_output_prob_log * \
        ((1 - c_switch).unsqueeze(2).expand_as(g_output_prob_log))

    g_output_prob_log = g_output_prob_log.view(-1, g_output_prob_log.size(2))
    c_output_prob_log = c_output_prob_log.view(-1, c_output_prob_log.size(2))
    # NOTICE !!!!! we can change how the loss is calculated.

    g_loss = crit(g_output_prob_log, g_targets.contiguous().view(-1))
    c_loss = copyCrit(c_output_prob_log, c_targets.contiguous().view(-1))
    total_loss = g_loss + c_loss
    return total_loss
