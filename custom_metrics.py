import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable
from typing import List, Optional, Tuple

from mmseg.registry import METRICS


@METRICS.register_module()
class MFNetMetric(BaseMetric):
    """mIoU metric that follows the original MFNet implementation.

    The original project accumulates a confusion matrix ignoring pixels whose
    label equals ``ignore_index`` and then computes:

    - overall accuracy using classes 1..N-1 (background excluded in the
      denominator).
    - per-class accuracies from the column-normalised confusion matrix.
    - IoU scores with the same formula as ``util.calculate_result`` in the
      reference codebase, where the background IoU (class id 0) stays at 0.
    - mean IoU is the plain average of the IoU vector (therefore including
      the background entry as zero), matching the training logs of the
      original MFNet repository.
    """

    default_prefix = 'mfnet'

    def __init__(self,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: List[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        for data_sample in data_samples:
            if 'pred_sem_seg' not in data_sample:
                continue
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            if 'gt_sem_seg' not in data_sample:
                continue
            label = data_sample['gt_sem_seg']['data'].squeeze().to(pred_label.device)
            valid = label != self.ignore_index
            if not torch.any(valid):
                continue

            pred = pred_label[valid].to(dtype=torch.int64)
            gt = label[valid].to(dtype=torch.int64)

            mask = (pred >= 0) & (pred < num_classes)
            mask &= (gt >= 0) & (gt < num_classes)
            if not torch.any(mask):
                continue

            pred = pred[mask].cpu().numpy()
            gt = gt[mask].cpu().numpy()

            conf = np.zeros((num_classes, num_classes), dtype=np.float64)
            np.add.at(conf, (gt, pred), 1)
            self.results.append(conf)

    def compute_metrics(self, results: List[np.ndarray]) -> dict:
        if len(results) == 0:
            return {}

        total_cf = np.sum(results, axis=0)
        overall_acc, acc, iou = self._calculate_result(total_cf)

        class_names = self.dataset_meta['classes']
        metrics = {
            'overall_acc': float(overall_acc),
            'mean_iou': float(iou.mean()),
        }
        for idx, cls_name in enumerate(class_names):
            metrics[f'acc_{cls_name}'] = float(acc[idx])
            metrics[f'iou_{cls_name}'] = float(iou[idx])

        logger = MMLogger.get_current_instance()
        summary_table = PrettyTable()
        summary_table.field_names = ['Metric', 'Value']
        summary_table.add_row(['overall_acc', f'{metrics["overall_acc"]:.4f}'])
        summary_table.add_row(['mean_iou', f'{metrics["mean_iou"]:.4f}'])

        class_table = PrettyTable()
        class_table.field_names = ['Class', 'Acc', 'IoU']
        for idx, cls_name in enumerate(class_names):
            class_table.add_row([
                cls_name,
                f'{acc[idx]:.4f}',
                f'{iou[idx]:.4f}',
            ])

        print_log('\n' + summary_table.get_string(), logger=logger)
        print_log('\n' + class_table.get_string(), logger=logger)
        return metrics

    @staticmethod
    def _calculate_result(cf: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        n_class = cf.shape[0]
        conf = np.zeros((n_class, n_class), dtype=np.float64)
        iou = np.zeros(n_class, dtype=np.float64)

        for cid in range(n_class):
            col_sum = cf[:, cid].sum()
            if col_sum > 0:
                conf[:, cid] = cf[:, cid] / col_sum

        for cid in range(1, n_class):
            union = cf[cid, 1:].sum() + cf[1:, cid].sum() - cf[cid, cid]
            if union > 0:
                iou[cid] = cf[cid, cid] / union

        denom = cf[1:, :].sum()
        overall_acc = 0.0
        if denom > 0:
            overall_acc = np.diag(cf[1:, 1:]).sum() / denom

        acc = np.diag(conf)
        return overall_acc, acc, iou
