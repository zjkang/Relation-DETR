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

    # 将分散的匹配索引转换为可以一次性索引的格式
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
            # gt_boxes: [(num_gt_boxes_i, 4)]
            # gt_labels: [(num_gt_boxes_i)]
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            # pred_boxes: (batch_size, num_queries, 4)
            # pred_logits: (batch_size, num_queries, num_classes)
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            # indices 是匈牙利匹配的结果，它是一个list，长度为batch_size，每个元素是一个tuple，
            # 包含两个tensor，分别表示预测和目标的匹配索引
            # 假设 batch_size = 2:
            # 第一张图片有3个gt boxes，num_queries = 100
            # 第二张图片有2个gt boxes，num_queries = 100
            # indices = [
            #     (
            #         tensor([45, 67, 89]),     # 预测框的索引 (从100个queries中选出的)
            #         tensor([0, 1, 2])         # 对应的gt框的索引 (总共3个gt)
            #     ),
            #     (
            #         tensor([23, 56]),         # 预测框的索引
            #         tensor([0, 1])            # 对应的gt框的索引 (总共2个gt)
            #     )
            # ]
            indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        return losses

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
        # 如果是分布式训练，需要汇总所有GPU上的目标框数量
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        # 将目标框数量除以GPU数量，并确保最小值为1
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # 移除aux_outputs和enc_outputs，只保留主要输出
        matching_outputs = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # 计算辅助损失 (来自decoder的每一层)
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # get matching results for each image
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes)
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})
        # 计算编码器输出的损失 (如果使用两阶段检测):
        # 遵循传统两阶段检测器的设计
        # 第一阶段（RPN）：只预测前景/背景
        # 第二阶段（RCNN）：进行具体的类别分类
        # 在计算loss时（通常在matcher中）
        # 前景的预测会与这个0标签计算loss
        # 背景的预测会与一个特殊的"no_object"类别计算loss
        # 通常"no_object"类别是最后一个类别索引（num_classes）
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            # 如果是二分类，将所有标签设为0（只区分前景/背景）
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes, is_encoder=True)
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        return losses


class HybridSetCriterion(SetCriterion):
    # Hybrid（混合）主要体现在它结合了两种不同的思想
    # 传统DETR的分类损失：
    # 使用one-hot标签
    # 基于匈牙利匹配的结果
    # 使用focal loss处理类别不平衡
    # 2. IoU-aware的分类损失（来自YOLO系列）：
    # 将IoU信息融入分类分数
    # 让分类分数同时反映定位质量
    # 使用IoU score作为软标签
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
        # 收集所有匹配到的ground truth的类别标签
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        # 在匹配位置填入真实类别标签
        target_classes[idx] = target_classes_o
        # num_classes + 1 是因为包含背景类
        # [..., :-1] 是去掉背景类
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        # 假设：
        # batch_size = 2
        # num_queries = 4
        # num_classes = 3
        # # target_classes 可能是：
        # [
        #     [3, 1, 3, 3],  # 第一张图：第2个query匹配到类别1，其他是背景(3)
        #     [0, 3, 2, 3]   # 第二张图：第1个query匹配到类别0，第3个匹配到类别2
        # ]
        # # target_classes_onehot 会变成：
        # [
        #     [[0,0,0], [0,1,0], [0,0,0], [0,0,0]],  # 第一张图
        #     [[1,0,0], [0,0,0], [0,0,1], [0,0,0]]   # 第二张图
        # ]

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


# class StableSetCriterion(SetCriterion):
#     def calculate_loss(self, outputs, targets, num_boxes, indices=None, is_encoder=False, layer_idx=None):
#         losses = {}
#         # get matching results for each image
#         if not indices:
#             # gt_boxes: [(num_gt_boxes_i, 4)]
#             # gt_labels: [(num_gt_boxes_i)]
#             gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
#             # pred_boxes: (batch_size, num_queries, 4)
#             # pred_logits: (batch_size, num_queries, num_classes)
#             pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
#             # indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels))
#             indices = [
#                 self.matcher(pb, pl, gb, gl,
#                             is_encoder=is_encoder,
#                             batch_idx=i,
#                             layer_idx=layer_idx)
#                 for i, (pb, pl, gb, gl) in enumerate(zip(pred_boxes, pred_logits, gt_boxes, gt_labels))
#             ]
#         loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
#         loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
#         losses.update(loss_class)
#         losses.update(loss_boxes)
#         return losses

#     def forward(self, outputs, targets):
#         """This performs the loss computation

#         :param outputs: dict of tensors, see the output specification of the model for the format
#         :param targets: list of dicts, such that len(targets) == batch_size
#         :return: a dict containing losses
#         """
#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor(
#             data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
#         )
#         # 如果是分布式训练，需要汇总所有GPU上的目标框数量
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         # 将目标框数量除以GPU数量，并确保最小值为1
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

#         if hasattr(self.matcher, 'reset_matches'):
#             self.matcher.reset_matches()

#         # Compute all the requested losses
#         losses = {}
#         # 移除aux_outputs和enc_outputs，只保留主要输出, 先处理最后一层（主输出），建立最终匹配目标
#         matching_outputs = {
#             k: v
#             for k, v in outputs.items()
#             if k != "aux_outputs" and k != "enc_outputs"
#         }
#         losses.update(self.calculate_loss(matching_outputs, targets, num_boxes,
#                                           is_encoder=False, layer_idx=None))

#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         # 从最后一个辅助层开始，向前处理
#         if "aux_outputs" in outputs:
#             # 计算辅助损失 (来自decoder的每一层)
#             num_layers = len(outputs["aux_outputs"]) # 辅助层数量 5
#             for i in range(num_layers - 1, -1, -1):  # 从后向前遍历
#                 aux_outputs = outputs["aux_outputs"][i]
#                 losses_aux = self.calculate_loss(
#                     aux_outputs, targets, num_boxes,
#                     is_encoder=False, layer_idx=i
#                 )
#                 losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

#         # 计算编码器输出的损失 (如果使用两阶段检测):
#         # 遵循传统两阶段检测器的设计
#         # 第一阶段（RPN）：只预测前景/背景
#         # 第二阶段（RCNN）：进行具体的类别分类
#         # 在计算loss时（通常在matcher中）
#         # 前景的预测会与这个0标签计算loss
#         # 背景的预测会与一个特殊的"no_object"类别计算loss
#         # 通常"no_object"类别是最后一个类别索引（num_classes）
#         if "enc_outputs" in outputs:
#             enc_outputs = outputs["enc_outputs"]
#             bin_targets = copy.deepcopy(targets)
#             # 如果是二分类，将所有标签设为0（只区分前景/背景）
#             if self.two_stage_binary_cls:
#                 for bt in bin_targets:
#                     bt["labels"] = torch.zeros_like(bt["labels"])
#             losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes, is_encoder=True)
#             losses.update({k + f"_enc": v for k, v in losses_enc.items()})

#         return losses


class StableHybridSetCriterion(SetCriterion):
    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict,
        alpha: float = 0.25,
        gamma: float = 2.0,
        two_stage_binary_cls=False,
        matching_copies=[(2,4),(2,4),(2,4),(2,4),(2,4),(2,4),(1,1)]
    ):
        """
        Args:
            matching_copies: 每层支持的many-to-one匹配数量, 例如[3,2,1,1,1]
            [el,dl_1,dl_2,dl_3,dl_4,dl_5,dl_6]
        """
        super().__init__(num_classes, matcher, weight_dict, alpha, gamma, two_stage_binary_cls)
        self.matching_copies = matching_copies

    def forward(self, outputs, targets):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            data=[num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # 如果是分布式训练，需要汇总所有GPU上的目标框数量
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        # 将目标框数量除以GPU数量，并确保最小值为1
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        # 移除aux_outputs和enc_outputs，只保留主要输出
        matching_outputs = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs" and k != "enc_outputs"
        }
        losses.update(self.calculate_loss(matching_outputs, targets, num_boxes, gt_copy=self.matching_copies[-1]))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # 计算辅助损失 (来自decoder的每一层)
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # get matching results for each image
                losses_aux = self.calculate_loss(aux_outputs, targets, num_boxes, gt_copy=self.matching_copies[i+1])
                losses.update({k + f"_{i}": v for k, v in losses_aux.items()})

        # 计算编码器输出的损失 (如果使用两阶段检测):
        # 遵循传统两阶段检测器的设计
        # 第一阶段（RPN）：只预测前景/背景
        # 第二阶段（RCNN）：进行具体的类别分类
        # 在计算loss时（通常在matcher中）
        # 前景的预测会与这个0标签计算loss
        # 背景的预测会与一个特殊的"no_object"类别计算loss
        # 通常"no_object"类别是最后一个类别索引（num_classes）
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            bin_targets = copy.deepcopy(targets)
            # 如果是二分类，将所有标签设为0（只区分前景/背景）
            if self.two_stage_binary_cls:
                for bt in bin_targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            losses_enc = self.calculate_loss(enc_outputs, bin_targets, num_boxes, gt_copy=self.matching_copies[0])
            losses.update({k + f"_enc": v for k, v in losses_enc.items()})

        return losses

    def calculate_loss(self, outputs, targets, num_boxes, indices=None, gt_copy=(1,1), **kwargs):
        losses = {}
        # get matching results for each image
        if not indices:
            # gt_boxes: [(num_gt_boxes_i, 4)]
            # gt_labels: [(num_gt_boxes_i)]
            gt_boxes, gt_labels = list(zip(*map(lambda x: (x["boxes"], x["labels"]), targets)))
            # pred_boxes: (batch_size, num_queries, 4)
            # pred_logits: (batch_size, num_queries, num_classes)
            pred_logits, pred_boxes = outputs["pred_logits"], outputs["pred_boxes"]
            # indices 是匈牙利匹配的结果，它是一个list，长度为batch_size，每个元素是一个tuple，
            # 包含两个tensor，分别表示预测和目标的匹配索引
            # 假设 batch_size = 2:
            # 第一张图片有3个gt boxes，num_queries = 100
            # 第二张图片有2个gt boxes，num_queries = 100
            # indices = [
            #     (
            #         tensor([45, 67, 89]),     # 预测框的索引 (从100个queries中选出的)
            #         tensor([0, 1, 2])         # 对应的gt框的索引 (总共3个gt)
            #     ),
            #     (
            #         tensor([23, 56]),         # 预测框的索引
            #         tensor([0, 1])            # 对应的gt框的索引 (总共2个gt)
            #     )
            # ]
            # indices = list(map(self.matcher, pred_boxes, pred_logits, gt_boxes, gt_labels, gt_copy))
            indices = list(map(
                lambda pb, pl, gb, gl: self.matcher(pb, pl, gb, gl, gt_copy=gt_copy[0], k=gt_copy[1]),
                pred_boxes, pred_logits, gt_boxes, gt_labels
            ))
        loss_class = self.loss_labels(outputs, targets, num_boxes, indices=indices)
        loss_boxes = self.loss_boxes(outputs, targets, num_boxes, indices=indices)
        losses.update(loss_class)
        losses.update(loss_boxes)
        return losses

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

        # construct onehot targets, target_classes_onehot shape: (batch_size, num_queries, num_classes)
        # 收集所有匹配到的ground truth的类别标签
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        # 在匹配位置填入真实类别标签
        target_classes[idx] = target_classes_o
        # num_classes + 1 是因为包含背景类
        # [..., :-1] 是去掉背景类
        target_classes_onehot = F.one_hot(target_classes, self.num_classes + 1)[..., :-1]

        # construct iou_score, shape: (batch_size, num_queries)
        target_score = torch.zeros_like(target_classes, dtype=iou_score.dtype)
        target_score[idx] = iou_score

        # 假设：
        # batch_size = 2
        # num_queries = 4
        # num_classes = 3
        # # target_classes 可能是：
        # [
        #     [3, 1, 3, 3],  # 第一张图：第2个query匹配到类别1，其他是背景(3)
        #     [0, 3, 2, 3]   # 第二张图：第1个query匹配到类别0，第3个匹配到类别2
        # ]
        # # target_classes_onehot 会变成：
        # [
        #     [[0,0,0], [0,1,0], [0,0,0], [0,0,0]],  # 第一张图
        #     [[1,0,0], [0,0,0], [0,0,1], [0,0,0]]   # 第二张图
        # ]

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