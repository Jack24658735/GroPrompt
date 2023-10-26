import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch.nn as nn
import math
import subprocess
import os
import pandas as pd
from mmengine.dist import is_main_process
import json

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

@HOOKS.register_module()
class CustomValidationHookA2D(Hook):
    """
    This hook will add the our custom validation.
    """
    priority = 'LOWEST' # NOTE: this must be later than the CheckpointHook

    def __init__(self):
        super(CustomValidationHookA2D, self).__init__()
        self.curr_epoch = 1
        self.best_epoch = 1
        
        self.best_score = 0
    
    # def before_run(self, runner) -> None:
    def after_train_epoch(self, runner) -> None:
        """This hook will add the our custom validation.

        Args:
            runner (Runner): The runner of the training process.
        """        
        infer_dir = os.path.join(runner.work_dir, f"epoch_{self.curr_epoch}")
        if is_main_process():
            cfg_path = runner.cfg.filename.split('/')[-1]
            
            cmd = [
                'python3',
                'inference_a2d_mmdet.py',
                '--binary',
                '--output_dir=' + infer_dir,
                '--dataset_file=a2d',
                '--online',
                '--num_frames=1',
                '--use_SAM',
                '--masks',
                # setup the model path for loading
                '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
                '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
                '--sam_ckpt_path=' + runner.cfg.sam_ckpt_path,
            ]
            # Run the command
            subprocess.run(cmd)
            ## TODO: add evaluation for on a2d or jhmdb
            # Read the JSON file
            with open(f'{infer_dir}/log.json', 'r') as f:
                eval_metric = json.load(f)

            # Extract the overall_iou value
            overall_iou = eval_metric.get('overall_iou', None)
            mean_iou = eval_metric.get('mean_iou', None)
            mAP = eval_metric.get('mAP 0.5:0.95', None)
            
            
            if mean_iou > self.best_score:
                self.best_score = mean_iou
                self.best_epoch = self.curr_epoch
            log_stats = { 
                        'overall_iou': overall_iou,
                        'mean_iou': mean_iou,
                        'mAP': mAP,
                        'best_epoch': self.best_epoch,
                        'best_score': self.best_score}
            print(log_stats)
            # TODO: save log outside at work_dir
            with open(os.path.join(runner.work_dir, 'our_log.txt'), 'a') as f:
                f.write(str(log_stats))
                f.write("\n")
            self.curr_epoch += 1
            
