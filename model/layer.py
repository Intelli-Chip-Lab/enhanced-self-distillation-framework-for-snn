# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron
import torch.nn.functional as F

# @torch.jit.script
def calcu_spikes_mean_and_var(spike: torch.Tensor):
    in_spikes_mean = spike.mean(dim=(0, 2, 3), keepdim=True)
    in_spikes_var = ((spike - in_spikes_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    return in_spikes_mean, in_spikes_var


# @torch.jit.script
def calcu_rate(x: torch.Tensor, in_spikes_mean: torch.Tensor, in_spikes_var: torch.Tensor, gamma: torch.Tensor,
               beta: torch.Tensor, eps: float):
    rate_mean = x.mean(dim=(0, 2, 3), keepdim=True)

    rate_mean = in_spikes_mean.detach() + (rate_mean - rate_mean.detach())

    rate_var = ((x - rate_mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)

    rate_var = in_spikes_var.detach() + (rate_var - rate_var.detach())
    rate_var = nn.functional.relu(rate_var)

    rate_hat = (x - rate_mean) / torch.sqrt((rate_var + eps))
    rate = gamma * rate_hat + beta
    return rate


class LIFLayer(neuron.LIFNode):

    def __init__(self, **cell_args):
        super(LIFLayer, self).__init__()
        tau = 1.0 / (1.0 - torch.sigmoid(cell_args['decay'])).item()
        super().__init__(tau=tau, decay_input=False, v_threshold=cell_args['thresh'], v_reset=cell_args['v_reset'],
                         detach_reset=cell_args['detach_reset'])
        self.register_memory('elig', 0.)
        self.register_memory('elig_factor', 0.)
        self.register_memory('out_spikes_mean', 0.)
        self.register_memory('current_step', 0)

    @staticmethod
    # @torch.jit.script
    def calcu_sg_and_elig(current_t: int, v: torch.Tensor, elig: torch.Tensor, elig_factor: float, v_threshold: float,
                          sigmoid_alpha: float = 4.0):
        sgax = ((v - v_threshold) * sigmoid_alpha).sigmoid()
        sg = (1. - sgax) * sgax * sigmoid_alpha
        elig = 1. / (current_t + 1) * (current_t * elig + elig_factor * sg)
        return elig

    @staticmethod
    # @torch.jit.script
    def calcu_elig(current_t: int, sg: torch.Tensor, pre_elig: torch.Tensor, pre_elig_factor: torch.Tensor):
        elig = 1. / (current_t + 1) * (current_t * pre_elig + pre_elig_factor * sg)
        return elig

    def calcu_elig_factor(self, elig_factor, lam, sg, spike):
        if self.v_reset is not None:  # hard-reset
            elig_factor = self.calcu_elig_factor_hard_reset(elig_factor, lam, spike, self.v, sg)
        else:  # soft-reset
            if not self.detach_reset:  # soft-reset w/ reset_detach==False
                elig_factor = self.calcu_elig_factor_soft_reset_not_detach_reset(elig_factor, lam, sg)

            else:  # soft-reset w/ reset_detach==True
                elig_factor = self.calcu_elig_factor_soft_reset_detach_reset(elig_factor, lam)

        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_hard_reset(elig_factor: torch.Tensor, lam: float, spike: torch.Tensor, v: torch.Tensor,
                                     sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam * (1. - spike) - lam * v * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_not_detach_reset(elig_factor: torch.Tensor, lam: float, sg: torch.Tensor):
        elig_factor = 1. + elig_factor * (lam - lam * sg)
        return elig_factor

    @staticmethod
    # @torch.jit.script
    def calcu_elig_factor_soft_reset_detach_reset(elig_factor: float, lam: float):
        elig_factor = 1. + elig_factor * lam
        return elig_factor

    def elig_init(self, x: torch.Tensor):
        self.elig = torch.zeros_like(x.data)
        self.elig_factor = 1.0

    def reset_state(self):
        self.reset()
        self.current_step = 0

    def forward(self, x, **kwargs):
        if not self.training :
            x = x.view(self.time_step, -1, *x.shape[1:])
            self.reset()
            # self.v = torch.zeros_like(x[0])
            spikes = []
            for t in range(self.time_step):
                self.v_float_to_tensor(x[t])
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                spikes.append(spike)
                self.neuronal_reset(spike)
            out = torch.cat(spikes, dim=0)
            self.spike = out
            return out
        elif self.training and self.spike_prop:
            x = x.view(self.time_step, -1, *x.shape[1:])
            self.reset()
            spikes = []
            lam = 1.0 - 1. / self.tau
            elig_factor = 1.0
            self.v_float_to_tensor(x[0])
            self.elig_init(x[0])
            for t in range(self.time_step):
                self.neuronal_charge(x[t])
                spike = self.neuronal_fire()
                self.elig = self.calcu_sg_and_elig(current_t=t, v=self.v, elig=self.elig, elig_factor=elig_factor,
                                              v_threshold=self.v_threshold)
                elig_factor = self.calcu_elig_factor_soft_reset_detach_reset(elig_factor, lam)
                spikes.append(spike)
                self.neuronal_reset(spike)
            out = torch.cat(spikes, dim=0)
            self.out_spikes_mean = out.view(self.time_step, -1, *out.shape[1:]).mean(dim=0)
            return out

        elif self.training and self.rate_prop:
            # assert len(x.shape) in (2,3,4)
            assert self.elig is not None and self.out_spikes_mean is not None
            rate = self.out_spikes_mean.detach() + (x * self.elig) - (x * self.elig).detach()
            return rate
        else:
            raise NotImplementedError()


def bn_forward_hook(module, args, output):

    if not module.training or hasattr(module, 'ann_branch'):
        return
    if not torch.is_grad_enabled():  # spike_prop stage
        in_spikes_mean, in_spikes_var = calcu_spikes_mean_and_var(args[0])

        module.in_spikes_mean = in_spikes_mean
        module.in_spikes_var = in_spikes_var

    else:  # rate_prop stage
        assert module.training and module.rate_prop

        rate = calcu_rate(args[0], module.in_spikes_mean, module.in_spikes_var,
                          gamma=module.weight.view(1, module.weight.shape[0], 1, 1),
                          beta=module.bias.view(1, module.bias.shape[0], 1, 1), eps=module.eps)

        module.track_running_stats = True
        return rate



def bn_forward_pre_hook(module, input):
    # testing stage or spike_prop stage
    if not module.training or module.spike_prop or hasattr(module, 'ann_branch'):
        return

    module.track_running_stats = False
    return

def esd_loss(logits_student, logits_teacher, temperature):
    valid_mask = (logits_teacher.abs().sum(dim=1) > 1e-6)  # shape: (batch_size,)

    # -----------------------------------------------
    logits_student = logits_student[valid_mask]
    logits_teacher = logits_teacher[valid_mask]

    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher.detach() / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd 


def make_teacher(avg_fr, labels):
    predictions = avg_fr.argmax(dim=2)  # shape (T, batch_size)

    correct_mask = (predictions == labels.unsqueeze(0))  # shape (T, batch_size)

    correct_avg_fr = avg_fr * correct_mask.unsqueeze(2)  # shape (T, batch_size, class_num)

    correct_count = correct_mask.sum(dim=0).unsqueeze(1)  # shape (batch_size, 1)
    epsilon = 1e-8
    correct_count = correct_count + epsilon
    teacher_labels = correct_avg_fr.sum(dim=0) / correct_count  # shape (batch_size, class_num)
    return teacher_labels



