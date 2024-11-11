import copy
from typing import Dict

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.losses import sigmoid_focal_loss, vari_sigmoid_focal_loss
from util.utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict,
        alpha: float = 0.25,
        gamma: float = 2.0,
        two_stage_binary_cls=False,
    ):
        """Create the criterion.

        :param num_classes: number of object categories, omitting the special no-object category
        :param matcher: module able to compute a matching between targets and proposals
        :param weight_dict: dict containing as key the names of the losses and as values their relative weight
        :param alpha: alpha in Focal Loss, defaults to 0.25
        :param gamma: gamma in Focal loss, defaults to 2.0
        :param two_stage_binary_cls: Whether to use two-stage binary classification loss, defaults to False
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.two_stage_binary_cls = two_stage_binary_cls

    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_class = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, targets, num_boxes, indices, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        return losses

    def predict_matching_targets(self, outputs, targets, layer_idx=None):
        gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
        pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]

        matching_stats = {}
        matching_stats.update(
            "matching_stats", self.analyze_matching(
                pred_logits, pred_boxes, gt_boxes, gt_labels))
        matching_stats.update(
            "query_predictions", self.analyze_query_predictions(
                pred_logits, top_k=5, num_display=10))
        return matching_stats

    def analyze_query_predictions(pred_logits, top_k=5, num_display=10):
        """分析query的预测情况
        Args:
            pred_logits: [B, N, num_classes]
            top_k: 每个query要显示的top k类别数
            num_display: 要显示的query数量
        """
        scores, classes = pred_logits.sigmoid().topk(top_k, dim=-1)  # [B, N, top_k]

        # 只选择置信度最高的num_display个query进行显示
        max_scores = scores[..., 0]  # [B, N]
        top_query_indices = max_scores.topk(num_display, dim=1)[1]  # [B, num_display]

        # 收集这些query的预测信息
        selected_scores = torch.gather(scores, 1,
            top_query_indices.unsqueeze(-1).expand(-1, -1, top_k))  # [B, num_display, top_k]
        selected_classes = torch.gather(classes, 1,
            top_query_indices.unsqueeze(-1).expand(-1, -1, top_k))  # [B, num_display, top_k]
        # 统计信息
        stats = {
            'mean_confidence': max_scores.mean().item(),
            'max_confidence': max_scores.max().item(),
            'min_confidence': max_scores.min().item(),
            'num_high_conf': (max_scores > 0.5).sum().item()  # 高置信度query的数量
        }
        return {
            'selected_queries': {
                'indices': top_query_indices,    # 选中的query索引
                'scores': selected_scores,       # 这些query的分数
                'classes': selected_classes      # 对应的类别
            },
            'stats': stats                      # 统计信息
        }

    def analyze_matching(
            self,
            pred_logits,
            pred_boxes,
            gt_boxes,
            gt_labels,
            iou_threshold=0.5):
        """分析每个GT被多少个预测匹配
        Args:
            pred_logits: [B, N, num_classes] - 预测类别logits
            pred_boxes: [B, N, 4] - 预测框
            gt_boxes: [B, M, 4] - GT框
            gt_labels: [B, M] - GT类别
            iou_threshold: float - IoU匹配阈值
        Returns:
            matching_stats: dict - 匹配统计信息
        """
        with torch.no_grad():
            batch_size = pred_boxes.shape[0]
            matching_stats = []

            # 获取预测类别
            pred_labels = pred_logits.argmax(dim=-1)  # [B, N]

            for b in range(batch_size):
                # 获取当前图片的GT
                cur_gt_boxes = gt_boxes[b]     # [M, 4]
                cur_gt_labels = gt_labels[b]    # [M]
                M = len(cur_gt_boxes)

                # 获取当前图片的预测
                cur_pred_boxes = pred_boxes[b]    # [N, 4]
                cur_pred_labels = pred_labels[b]  # [N]

                # 计算IoU矩阵
                iou_matrix = box_ops.box_iou(cur_gt_boxes, cur_pred_boxes)  # [M, N]

                # 统计每个GT匹配的预测数量
                matched_preds_per_gt = []
                for gt_idx in range(M):
                    # 找到IoU大于阈值且类别匹配的预测
                    matched_mask = (iou_matrix[gt_idx] > iou_threshold) & (cur_pred_labels == cur_gt_labels[gt_idx])
                    num_matches = matched_mask.sum().item()

                    matched_preds_per_gt.append({
                        'gt_idx': gt_idx,
                        'gt_label': cur_gt_labels[gt_idx].item(),
                        'num_matches': num_matches,
                        'matched_pred_indices': matched_mask.nonzero().squeeze(-1).tolist(),
                        'matched_ious': iou_matrix[gt_idx][matched_mask].tolist()
                    })

                matching_stats.append({
                    'image_id': b,
                    'num_gts': M,
                    'matches_per_gt': matched_preds_per_gt,
                    'total_matches': sum(m['num_matches'] for m in matched_preds_per_gt)
                })

            # 计算整体统计信息
            total_gts = sum(stat['num_gts'] for stat in matching_stats)
            total_matches = sum(stat['total_matches'] for stat in matching_stats)
            avg_matches_per_gt = total_matches / total_gts if total_gts > 0 else 0

            summary = {
                'total_gts': total_gts,
                'total_matches': total_matches,
                'avg_matches_per_gt': avg_matches_per_gt,
                'per_image_stats': matching_stats
            }
            return summary


    def forward(self, outputs, targets):
        """This performs the loss computation

        :param outputs: dict of tensors, see the output specification of the model for the format
        :param targets: list of dicts, such that len(targets) == batch_size
        :return: a dict containing losses
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        matching_outputs = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # get matching results for each image
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes)
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes)
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        matching_stats_dict = {}
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                matching_stats = self.predict_matching_targets(aux_outputs, targets, layer_idx=i)
                matching_stats_dict.update({f"matching_stats_{i}": matching_stats})
        final_layer_idx = len(outputs["aux_outputs"])
        matching_stats = self.predict_matching_targets(matching_outputs, targets, final_layer_idx)
        matching_stats_dict.update({f"matching_stats_{final_layer_idx}": matching_stats})

        return losses, matching_stats_dict


class HybridSetCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, num_boxes, indices, **kwargs):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        iou_score = torch.diag(
            box_ops.box_iou(
                box_ops._box_cxcywh_to_xyxy(src_boxes),
                box_ops._box_cxcywh_to_xyxy(target_boxes),
            )
        ).detach()  # add detach according to RT-DETR

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # construct onehot targets, shape: (batch_size, num_queries, num_classes)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        loss_class = (
            vari_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                target_score,
                num_boxes=num_boxes,
                alpha=self.alpha,
                gamma=self.gamma,
            ) * src_logits.shape[1]
        )
        losses = {"loss_class": loss_class}
        return losses
