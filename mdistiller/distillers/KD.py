import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def sort(logit, target):
    """Sorting algorithm"""
    gt_mask = torch.zeros_like(logit).scatter_(1, target.unsqueeze(1), 1).bool()
    logit1 = torch.clone(logit)
    #modified logit
    logit1 = logit + (gt_mask * 1e9)

    #sort logits
    _, sorted_t_indices = torch.sort(logit1, dim=1, descending=True)
    sorted_t_logits, _ = torch.sort(logit, dim=1, descending=True)

    #unsort
    unsorted_logits = torch.zeros_like(sorted_t_logits)
    unsorted_logits.scatter_(1, sorted_t_indices, sorted_t_logits)
    return unsorted_logits

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class KD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(KD, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss(
            logits_student, sort(logits_teacher, target), self.temperature
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
