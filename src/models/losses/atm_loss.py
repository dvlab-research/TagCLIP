import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES
from .criterion import SegPlusCriterion

@LOSSES.register_module()
class SegLossPlus(nn.Module):
    """ATMLoss.
    """
    def __init__(self,
                 num_classes,
                 dec_layers,
                 mask_weight=20.0,
                 dice_weight=1.0,
                 disc_weight=1.0,
                 loss_weight=1.0,
                 loss_type='dice',
                 use_point=False):
        super(SegLossPlus, self).__init__()
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight, 'loss_disc': disc_weight}
        aux_weight_dict = {}
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        self.criterion = SegPlusCriterion(
            num_classes,
            weight_dict=weight_dict,
            losses=["masks"],
            loss_type=loss_type,
        )

        self.loss_weight = loss_weight

    def forward(self,
                outputs,
                label,
                ignore_index=255,
                ):
        """Forward function."""
        
        self.ignore_index = ignore_index
        targets = self.prepare_targets(label)
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] = losses[k] * self.criterion.weight_dict[k] * self.loss_weight
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        return losses

    def prepare_targets(self, targets_dict):
        new_targets = []
        if isinstance(targets_dict, dict):
            targets = targets_dict['gt_semantic_seg']
            targets_unmasked = targets_dict['gt_semantic_seg_unmasked']
            disc_target = targets_dict['discriminate_map']

            for target_unmasked, target in zip(targets_unmasked, targets):
                gt_cls = target.unique()
                gt_cls = gt_cls[gt_cls != self.ignore_index]
                masks = []
                for cls in gt_cls:
                    masks.append(target == cls)
                if len(gt_cls) == 0:
                    masks.append(target == self.ignore_index)
                masks = torch.stack(masks, dim=0)

                new_targets.append(
                    {
                        "labels": gt_cls,
                        "target_masks": masks,
                        'masks_unmasked': target_unmasked,
                        "masks": target,
                        'disc_target': disc_target,
                    }
                )
            return new_targets

        else:
            targets = targets_dict
            for target in targets:
                gt_cls = target.unique()
                gt_cls = gt_cls[gt_cls != self.ignore_index]
                masks = []
                for cls in gt_cls:
                    masks.append(target == cls)
                if len(gt_cls) == 0:
                    masks.append(target == self.ignore_index)
                masks = torch.stack(masks, dim=0)
                new_targets.append(
                    {
                        "labels": gt_cls,
                        "target_masks": masks,
                        "masks": target,
                    }
                )
            return new_targets
