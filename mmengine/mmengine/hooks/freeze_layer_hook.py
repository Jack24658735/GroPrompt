import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class FreezeLayerHook(Hook):
    """
    This hook will freeze the layers of the model.
    """
    priority = 'NORMAL'

    def __init__(self, dummy=None):
        self.dummy = dummy

    def before_run(self, runner) -> None:
        """Freeze the layers of the model.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        for param in model.parameters():
            param.requires_grad = False
        # Enable transformer decoder
        for param in model.decoder.parameters():
            param.requires_grad = True
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        print('number of params for tuning:', n_parameters)
        print('Total number of params for original model:', total_parameters)
