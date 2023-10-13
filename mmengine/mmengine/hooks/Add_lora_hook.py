import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch.nn as nn
import math


class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x

@HOOKS.register_module()
class AddLoRAHook(Hook):
    """
    This hook will add the LORA layers to the model.
    """
    priority = 'LOW'

    def __init__(self):
        super(AddLoRAHook, self).__init__()
    
    def before_train(self, runner) -> None:
        """Add LORA layers to the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        r = 4 # LORA rank (can be modified)
        assert r > 0
        dim = model.decoder.layers[0].cross_attn.value_proj.in_features

        for param in model.parameters():
            param.requires_grad = False
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        for layer in model.decoder.layers:
            # layer.cross_attn.value_proj
            # layer.cross_attn.output_proj
            w_q_linear = layer.cross_attn.value_proj
            w_v_linear = layer.cross_attn.output_proj
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            layer.cross_attn.value_proj = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            layer.cross_attn.output_proj = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)
        
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.kaiming_uniform_(w_B.weight, a=math.sqrt(5))
            # nn.init.zeros_(w_B.weight)
        model = model.cuda()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        print('number of params for LORA tuning:', n_parameters)
        print('Total number of params for original model:', total_parameters)
