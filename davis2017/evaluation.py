import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from davis2017.davis import DAVIS
from davis2017.metrics import db_eval_boundary, db_eval_iou
from davis2017 import utils
from davis2017.results import Results
from scipy.optimize import linear_sum_assignment
import csv
import os


class DAVISEvaluation(object):
    def __init__(self, davis_root, task, gt_set, sequences='all', codalab=False):
        """
        Class to evaluate DAVIS sequences from a certain set and for a certain task
        :param davis_root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to compute the evaluation, chose between semi-supervised or unsupervised.
        :param gt_set: Set to compute the evaluation
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.davis_root = davis_root
        self.task = task
        self.dataset = DAVIS(root=davis_root, task=task, subset=gt_set, sequences=sequences, codalab=codalab)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    @staticmethod
    def _evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric, max_n_proposals=20):
        if all_res_masks.shape[0] > max_n_proposals:
            sys.stdout.write(f"\nIn your PNG files there is an index higher than the maximum number ({max_n_proposals}) of proposals allowed!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        f_metrics_res = np.zeros((all_res_masks.shape[0], all_gt_masks.shape[0], all_gt_masks.shape[1]))
        for ii in range(all_gt_masks.shape[0]):
            for jj in range(all_res_masks.shape[0]):
                if 'J' in metric:
                    j_metrics_res[jj, ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
                if 'F' in metric:
                    f_metrics_res[jj, ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[jj, ...], all_void_masks)
        if 'J' in metric and 'F' in metric:
            all_metrics = (np.mean(j_metrics_res, axis=2) + np.mean(f_metrics_res, axis=2)) / 2
        else:
            all_metrics = np.mean(j_metrics_res, axis=2) if 'J' in metric else np.mean(f_metrics_res, axis=2)
        row_ind, col_ind = linear_sum_assignment(-all_metrics)
        return j_metrics_res[row_ind, col_ind, :], f_metrics_res[row_ind, col_ind, :]

    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            if self.task == 'unsupervised':
                j_metrics_res, f_metrics_res = self._evaluate_unsupervised(all_gt_masks, all_res_masks, all_void_masks, metric)
            elif self.task == 'semi-supervised':
                j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                # if ii >= 3 and seq == 'india':
                #     break
                
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM
            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res
    
    @staticmethod
    def masks_to_bboxes(masks):
        """
        Convert binary masks to bounding boxes.

        Args:
        - masks (numpy.ndarray): Binary masks with shape (num_objects, video_length, height, width).

        Returns:
        - bboxes (numpy.ndarray): Bounding boxes with shape (num_objects, video_length, 4).
        """
        num_objects, video_length, _, _ = masks.shape

        # Initialize bounding boxes
        bboxes = np.zeros((num_objects, video_length, 4))

        for obj_idx in range(num_objects):
            for frame_idx in range(video_length):
                # Get the current mask
                mask = masks[obj_idx, frame_idx]

                # Check if the mask is not empty
                if np.any(mask):
                    rows = np.any(mask, axis=1)
                    cols = np.any(mask, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    bboxes[obj_idx, frame_idx] = [cmin, rmin, cmax, rmax]
                else:
                    bboxes[obj_idx, frame_idx] = [0, 0, 0, 0]  # Default to an empty bounding box

        return bboxes
    
    @staticmethod
    def load_csv_bboxes(file_path, num_objects, video_length):
        """
        Load bounding boxes from the CSV file and format them to match the shape of the masks_to_bboxes output.

        Args:
        - file_path (str): Path to the CSV file.
        - num_objects (int): Number of distinct objects.
        - video_length (int): Number of frames in the video.

        Returns:
        - bbox_data (numpy.ndarray): Bounding boxes with shape (num_objects, video_length, 4).
        """
        # Initialize empty array
        bbox_data = np.zeros((num_objects, video_length, 4), dtype=np.int32)

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header row

            for row in reader:
                obj_id = int(row[2]) - 1  # Subtracting 1 because Python index starts from 0
                frame_idx = int(row[1].split('.')[0])  # Assuming frame filenames are like "00000.jpg", "00001.jpg", etc.
                bbox = [
                    float(row[3].replace("tensor(", "").replace(")", "")),
                    float(row[4].replace("tensor(", "").replace(")", "")),
                    float(row[5].replace("tensor(", "").replace(")", "")),
                    float(row[6].replace("tensor(", "").replace(")", ""))
                ]

                bbox_data[obj_id, frame_idx] = bbox

        return bbox_data

    @staticmethod
    def compute_iou(gt_bboxes, pred_bboxes):
        """
        Compute the Intersection over Union (IoU) between ground truth and predicted bounding boxes.

        Args:
        - gt_bboxes (numpy.ndarray): Ground truth bounding boxes with shape (num_objects, video_length, 4).
        - pred_bboxes (numpy.ndarray): Predicted bounding boxes with the same shape.

        Returns:
        - ious (numpy.ndarray): IoUs with shape (num_objects, video_length).
        """
        # Determine the coordinates of the intersection rectangle
        x1_inter = np.maximum(gt_bboxes[..., 0], pred_bboxes[..., 0])
        y1_inter = np.maximum(gt_bboxes[..., 1], pred_bboxes[..., 1])
        x2_inter = np.minimum(gt_bboxes[..., 2], pred_bboxes[..., 2])
        y2_inter = np.minimum(gt_bboxes[..., 3], pred_bboxes[..., 3])

        # Compute the area of intersection rectangle
        inter_area = np.maximum(0, x2_inter - x1_inter + 1) * np.maximum(0, y2_inter - y1_inter + 1)

        # Compute the area of both GT and predicted bounding boxes
        gt_bbox_area = (gt_bboxes[..., 2] - gt_bboxes[..., 0] + 1) * (gt_bboxes[..., 3] - gt_bboxes[..., 1] + 1)
        pred_bbox_area = (pred_bboxes[..., 2] - pred_bboxes[..., 0] + 1) * (pred_bboxes[..., 3] - pred_bboxes[..., 1] + 1)

        # Compute IoU
        iou = inter_area / (gt_bbox_area + pred_bbox_area - inter_area)

        return iou
    

    def evaluate_bbox(self, res_path):
        results = []
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, _, all_masks_id = self.dataset.get_all_masks(seq, True)
            all_gt_bboxes = self.masks_to_bboxes(all_gt_masks)
            num_objs, video_len = all_gt_bboxes.shape[0], all_gt_bboxes.shape[1]
            all_pred_bboxes = self.load_csv_bboxes(os.path.join(res_path, seq, 'bounding_box_data.csv'), num_objs, video_len)
            ious = self.compute_iou(all_gt_bboxes, all_pred_bboxes)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                mIoU = np.mean(ious[ii])
                results.append({
                    "Sequence": seq_name,
                    "m_iou": mIoU
                })
        return results