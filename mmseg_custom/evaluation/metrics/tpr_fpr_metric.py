# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.registry import METRICS
from .iou_metric import IoUMetric


@METRICS.register_module()
class TPRFPRMetric(IoUMetric):
    """IoU metric extended with TPR (True Positive Rate) and FPR (False
    Positive Rate) per class.

    TPR = TP / (TP + FN)  (= Recall = Sensitivity)
    FPR = FP / (FP + TN)
    """

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {self.output_dir}')
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        # Standard metrics (IoU, Acc, etc.)
        ret_metrics = self.total_area_to_metrics(
            total_area_intersect, total_area_union, total_area_pred_label,
            total_area_label, self.metrics, self.nan_to_num, self.beta)

        # --- TPR / FPR computation ---
        # Per class: TP, FP, FN, TN
        tp = total_area_intersect.numpy()
        fp = (total_area_pred_label - total_area_intersect).numpy()
        fn = (total_area_label - total_area_intersect).numpy()
        total_pixels = total_area_label.sum().numpy()
        tn = total_pixels - tp - fp - fn

        tpr = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        fpr = np.where((fp + tn) > 0, fp / (fp + tn), 0.0)

        ret_metrics['TPR'] = tpr
        ret_metrics['FPR'] = fpr

        class_names = self.dataset_meta['classes']

        # Summary table (mean values)
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        metrics = dict()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                metrics[key] = val
            else:
                metrics['m' + key] = val

        # Per-class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)

        return metrics
