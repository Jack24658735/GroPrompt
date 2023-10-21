import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
import torch.nn as nn
import math
import subprocess
import os
import pandas as pd
from mmengine.dist import is_main_process


@HOOKS.register_module()
class CustomValidationHook(Hook):
    """
    This hook will add the our custom validation.
    """
    priority = 'LOWEST' # NOTE: this must be later than the CheckpointHook

    def __init__(self):
        super(CustomValidationHook, self).__init__()
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
        anno0_dir = os.path.join(infer_dir, "valid", "anno_0")
        if is_main_process():
            d_lora_hook = {'type': 'AddLoRAHook'}
            cfg_path = runner.cfg.filename.split('/')[-1]
            if d_lora_hook in runner.cfg['custom_hooks']:
                cmd = [
                    'python3',
                    'inference_davis_online_dino_mmdet.py',
                    '--binary',
                    '--output_dir=' + infer_dir,
                    '--dataset_file=davis',
                    '--online',
                    '--visualize',
                    '--use_trained_gdino',
                    # setup the model path for loading
                    '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
                    '--use_gdino_LORA',
                    '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
                    
                ]
            else:
                cmd = [
                    'python3',
                    'inference_davis_online_dino_mmdet.py',
                    '--binary',
                    '--output_dir=' + infer_dir,
                    '--dataset_file=davis',
                    '--online',
                    '--visualize',
                    '--use_trained_gdino',
                    # setup the model path for loading
                    '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
                    '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
                ]
            # Run the command
            subprocess.run(cmd)
            ## TODO: add evaluation for DAVIS
            # Passing arguments to the bash script
            cmd_eval = [
                'python3',
                'eval_davis.py',
                '--results_path=' + anno0_dir,
                '--eval_bbox'
            ]
            subprocess.run(cmd_eval)

            # Obtain iou score
            file_path = os.path.join(anno0_dir, 'bbox_results-val.csv')

            # Read the CSV file into a DataFrame
            df_iou = pd.read_csv(file_path)

            # Fetch the value corresponding to "Global"
            iou_score = df_iou[df_iou['Sequence'] == 'Global']['m_iou'].iloc[0]
            
            if iou_score > self.best_score:
                self.best_score = iou_score
                self.best_epoch = self.curr_epoch
            log_stats = { 
                        'iou_score': iou_score, 
                        'best_epoch': self.best_epoch,
                        'best_score': self.best_score}
            print(log_stats)
            # TODO: save log outside at work_dir
            with open(os.path.join(runner.work_dir, 'our_log.txt'), 'a') as f:
                f.write(str(log_stats))
                f.write("\n")
            self.curr_epoch += 1
