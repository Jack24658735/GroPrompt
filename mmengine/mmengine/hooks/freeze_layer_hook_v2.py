import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class FreezeLayerHookV2(Hook):
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
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        for param in model.parameters():
            param.requires_grad = False
        # Enable transformer decoder
        for param in model.decoder.parameters():
            param.requires_grad = True
        # Enable reg branch in bbox head
        for param in model.bbox_head.reg_branches.parameters():
            param.requires_grad = True
        # TODO: enable MLP for video loss
        for param in model.bbox_head.sam_image_proj_head.parameters():
            param.requires_grad = True
        for param in model.bbox_head.sam_prompt_proj_head.parameters():
            param.requires_grad = True
        for param in model.bbox_head.video_attn.parameters():
            param.requires_grad = True
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        print('number of params for tuning:', n_parameters)
        print('Total number of params for original model:', total_parameters)
