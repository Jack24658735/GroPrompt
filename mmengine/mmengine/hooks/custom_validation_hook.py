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
        anno1_dir = os.path.join(infer_dir, "valid", "anno_1")
        anno2_dir = os.path.join(infer_dir, "valid", "anno_2")
        anno3_dir = os.path.join(infer_dir, "valid", "anno_3")
        if is_main_process():
            d_lora_hook = {'type': 'AddLoRAHook'}
            cfg_path = runner.cfg.filename.split('/')[-1]
            # if d_lora_hook in runner.cfg['custom_hooks']:
            #     cmd = [
            #         'python3',
            #         'inference_davis_online_dino_mmdet.py',
            #         '--binary',
            #         '--output_dir=' + infer_dir,
            #         '--dataset_file=davis',
            #         '--online',
            #         '--visualize',
            #         '--use_trained_gdino',
            #         # setup the model path for loading
            #         '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
            #         '--use_gdino_LORA',
            #         '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
            #         '--ngpu=' + '4',
                    
            #     ]
            # else:
            #     cmd = [
            #         'python3',
            #         'inference_davis_online_dino_mmdet.py',
            #         '--binary',
            #         '--output_dir=' + infer_dir,
            #         '--dataset_file=davis',
            #         '--online',
            #         '--visualize',
            #         '--use_trained_gdino',
            #         # setup the model path for loading
            #         '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
            #         '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
            #         '--ngpu=' + '4',
            #     ]
            if d_lora_hook in runner.cfg['custom_hooks']:
                cmd = [
                    'python3',
                    'inference_davis_online_sam_mmdet.py',
                    '--binary',
                    '--output_dir=' + infer_dir,
                    '--dataset_file=davis',
                    '--online',
                    '--use_SAM',
                    '--visualize',
                    # setup the model path for loading
                    '--sam_ckpt_path=' + f'sam_hq_vit_h.pth',
                    '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
                    '--use_gdino_LORA',
                    '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
                    '--ngpu=' + '8',
                    '--run_anno_id=' + '4'
                    
                ]
            else:
                cmd = [
                    'python3',
                    'inference_davis_online_sam_mmdet.py',
                    '--binary',
                    '--output_dir=' + infer_dir,
                    '--dataset_file=davis',
                    '--online',
                    '--use_SAM',
                    '--visualize',
                    '--sam_ckpt_path=' + f'sam_hq_vit_h.pth',
                    # setup the model path for loading
                    '--g_dino_ckpt_path=' + f'{runner.work_dir}/epoch_{self.curr_epoch}.pth',
                    '--g_dino_config_path=' + f'{runner.work_dir}/{cfg_path}',
                    '--ngpu=' + '8',
                    '--run_anno_id=' + '4'

                ]
            # Run the command
            subprocess.run(cmd)
            ## TODO: add evaluation for DAVIS
            # Passing arguments to the bash script
            j_f_score_list = []
            iou_score_list = []
            for anno_dir in [anno0_dir, anno1_dir, anno2_dir, anno3_dir]:
                cmd_eval = [
                    'python3',
                    'eval_davis.py',
                    '--results_path=' + anno_dir,
                    '--eval_bbox',
                    '--eval_mask',
                ]
                subprocess.run(cmd_eval)

                # Obtain iou score
                file_path = os.path.join(anno_dir, 'bbox_results-val.csv')

                # Read the CSV file into a DataFrame
                df_iou = pd.read_csv(file_path)

                # Fetch the value corresponding to "Global"
                iou_score = df_iou[df_iou['Sequence'] == 'Global']['m_iou'].iloc[0]

                file_mask_path = os.path.join(anno_dir, 'global_results-val.csv')
                df_mask = pd.read_csv(file_mask_path)

                # Extract the value for 'J&F-Mean'
                j_and_f_mean = df_mask['J&F-Mean'].values[0]

                j_f_score_list.append(j_and_f_mean)
                iou_score_list.append(iou_score)

                # total_j_f_score += j_f_score
            avg_jf_score = sum(j_f_score_list) / 4
            avg_bbox_score = sum(iou_score_list) / 4
            
            if avg_jf_score > self.best_score:
                self.best_score = avg_jf_score
                self.best_epoch = self.curr_epoch
            log_stats = { 
                        'iou_score': avg_bbox_score, 
                        'anno0_iou_score': iou_score_list[0],
                        'anno1_iou_score': iou_score_list[1],
                        'anno2_iou_score': iou_score_list[2],
                        'anno3_iou_score': iou_score_list[3],
                        'best_epoch': self.best_epoch,
                        'best_score': self.best_score,
                        'j&f_score': avg_jf_score,
                        'anno0_j&f_score': j_f_score_list[0],
                        'anno1_j&f_score': j_f_score_list[1],
                        'anno2_j&f_score': j_f_score_list[2],
                        'anno3_j&f_score': j_f_score_list[3],
                        }
            print(log_stats)
            # TODO: save log outside at work_dir
            with open(os.path.join(runner.work_dir, 'our_log.txt'), 'a') as f:
                f.write(str(log_stats))
                f.write("\n")
            self.curr_epoch += 1
