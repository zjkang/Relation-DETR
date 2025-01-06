import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""
    def __init__(
        self,
        select_box_nums_for_evaluation=100,
        nms_iou_threshold=-1,
        confidence_score=-1,
    ):
        super().__init__()
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_score = confidence_score

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        # 获取模型输出
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # 获取每个query的置信度
        prob = out_logits.sigmoid()
        # 选择top-k个最高置信度的预测
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1),
            self.select_box_nums_for_evaluation,
            dim=1,
        )
        # [batch_size, 100]
        scores = topk_values
        # 取对应的类别和框
        # 假设:
        # out_logits.shape = [batch_size, num_queries=100, num_classes=20]
        # topk_indexes的形状是 [batch_size, k]，包含展平后的索引

        # 例如，对于一个具体的索引值 index = 1234:
        # - 这个索引来自将 [num_queries, num_classes] 展平成一维
        # - 需要还原出这个索引对应的 query_id 和 class_id
        topk_boxes = torch.div(topk_indexes, out_logits.shape[2], rounding_mode="trunc") # 获取query的索引
        labels = topk_indexes % out_logits.shape[2] # 获取类别的索引
        boxes = box_ops._box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        item_indice = None
        # filter low-confidence predictions
        if self.confidence_score > 0:
            # 生成布尔掩码，只保留置信度高于阈值的预测
            item_indice = [score > self.confidence_score for score in scores]

        # filter overlap predictions NMS (Non-Maximum Suppression) 过滤
        if self.nms_iou_threshold > 0:
            # 对每张图片分别进行NMS
            nms_indice = [
                box_ops.nms(box, score, iou_threshold=self.nms_iou_threshold)
                for box, score in zip(boxes, scores)
            ]
            # 将NMS结果转换为布尔掩码
            nms_binary_indice = [torch.zeros_like(item_index, dtype=torch.bool) for item_index in item_indice]
            for nms_binary_index, nms_index in zip(nms_binary_indice, nms_indice):
                nms_binary_index[nms_index] = True
            # 结合置信度过滤和NMS过滤的结果
            item_indice = [
                item_index & nms_binary_index
                for item_index, nms_binary_index in zip(item_indice, nms_binary_indice)
            ]

        # 应用过滤结果
        if item_indice is not None:
            # 只保留通过所有过滤的预测
            scores = [score[item_index] for score, item_index in zip(scores, item_indice)]
            boxes = [box[item_index] for box, item_index in zip(boxes, item_indice)]
            labels = [label[item_index] for label, item_index in zip(labels, item_indice)]

        if torchvision._is_tracing():
            # avoid interation warning during ONNX export
            scores, labels, boxes = map(lambda x: x.unbind(0), (scores, labels, boxes))
        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results


class SegmentationPostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes, input_sizes, batched_input_size):
        out_logits, out_bbox, out_mask = (
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_masks"],
        )

        assert len(out_logits) == len(target_sizes)
        assert len(batched_input_size) == 2

        # we average queries of the same class to get onehot segmentation image
        out_class = out_logits.argmax(-1)
        num_class = out_logits.shape[-1]
        result_masks = []
        for image_id in range(len(out_logits)):
            result_masks_per_image = []
            for cur_class in range(num_class):
                class_index = out_class[image_id] == cur_class
                mask_per_class = out_mask[image_id][class_index].sigmoid()
                if mask_per_class.numel() == 0:
                    mask_per_class = mask_per_class.new_zeros((1, *mask_per_class.shape[-2:]))
                mask_per_class = mask_per_class.mean(0)
                result_masks_per_image.append(mask_per_class)
            result_masks_per_image = torch.stack(result_masks_per_image, 0)
            result_masks.append(result_masks_per_image)
        result_masks = torch.stack(result_masks, 0)

        # upsample masks with 1/4 resolution to input image shapes
        result_masks = F.interpolate(
            result_masks,
            size=batched_input_size,
            mode="bilinear",
            align_corners=False,
        )

        # resize masks to original shapes and transform onehot into class
        mask_results = []
        for mask, (height, width), (out_height, out_width) in zip(
            result_masks,
            input_sizes,
            target_sizes,
        ):
            mask = F.interpolate(
                mask[None, :, :height, :width],
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            mask_results.append({"masks": mask.argmax(0)})

        return mask_results
