from abc import ABC, abstractmethod
from .layer import *


class RateModel(nn.Module):

    def wrap_model(self, **kwargs):
        assert ('time_step' in kwargs and kwargs.get(
            'time_step') > 0)
        time_step = kwargs.get('time_step')

        for name, module in self.named_modules():
            setattr(module, 'time_step', time_step)
        for name, module in self.named_modules():
            if isinstance(module,  (nn.BatchNorm2d,)):
                module.register_forward_pre_hook(bn_forward_pre_hook)
                module.register_forward_hook(bn_forward_hook)

    def reset_model(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.BatchNorm2d,)):
                module.in_spikes = []
                module.in_spikes_mean = None
                module.in_spikes_var = None
            elif isinstance(module, neuron.LIFNode):
                module.reset_state()
            else:
                continue

    def __set_prop_state(self, spike_prop_state, rate_prop_state):
        for module in self.modules():
            module.spike_prop = spike_prop_state
            module.rate_prop = rate_prop_state

    def set_spike_prop_state(self):
        self.__set_prop_state(True, False)

    def set_rate_prop_state(self):
        self.__set_prop_state(False, True)
