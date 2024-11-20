import torch
from torch import nn
from torchvision.ops import boxes as box_ops

from util.misc import inverse_sigmoid


class GenerateDNQueries(nn.Module):
    """Generate denoising queries for DN-DETR

    Args:
        num_queries (int): Number of total queries in DN-DETR. Default: 300
        num_classes (int): Number of total categories. Default: 80.
        label_embed_dim (int): The embedding dimension for label encoding. Default: 256.
        denoising_groups (int): Number of noised ground truth groups. Default: 5.
        label_noise_prob (float): The probability of the label being noised. Default: 0.2.
        box_noise_scale (float): Scaling factor for box noising. Default: 0.4
        with_indicator (bool): If True, add indicator in noised label/box queries.

    """
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_groups: int = 5,
        label_noise_prob: float = 0.2,
        box_noise_scale: float = 0.4,
        with_indicator: bool = False,
    ):
        super(GenerateDNQueries, self).__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.label_embed_dim = label_embed_dim
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.with_indicator = with_indicator

        # leave one dim for indicator mentioned in DN-DETR
        if with_indicator:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim - 1)
        else:
            self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    @staticmethod
    def apply_label_noise(labels: torch.Tensor, label_noise_prob: float = 0.2, num_classes: int = 80):
        """给标签添加噪声
        Args:
            labels: [N] - 原始标签
            label_noise_prob: float - 添加噪声的概率
            num_classes: int - 类别总数
        Returns:
            noised_labels: [N] - 添加噪声后的标签
        """
        if label_noise_prob > 0:
            # 1. 生成随机mask，决定哪些位置需要添加噪声
            mask = torch.rand_like(labels.float()) < label_noise_prob
            # 例如：labels = [1, 2, 3], mask可能是[True, False, True]
            # 2. 生成随机标签
            noised_labels = torch.randint_like(labels, 0, num_classes)
            # 例如：noised_labels可能是[5, 7, 2]
            # 3. 将需要添加噪声的位置替换为随机标签
            noised_labels = torch.where(mask, noised_labels, labels)
            # 如果mask是[True, False, True]
            # labels是[1, 2, 3]
            # noised_labels是[5, 7, 2]
            # 最终结果是[5, 2, 2]  # True位置用noised_labels，False位置用labels
            return noised_labels
        else:
            return labels # 如果noise_prob=0，直接返回原始标签

    @staticmethod
    def apply_box_noise(boxes: torch.Tensor, box_noise_scale: float = 0.4):
        """给边界框添加噪声
        Args:
            boxes: [N, 4] - 原始边界框
            box_noise_scale: float - 边界框噪声的缩放因子
        Returns:
            noised_boxes: [N, 4] - 添加噪声后的边界框
        """
        if box_noise_scale > 0:
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2
            diff[:, 2:] = boxes[:, 2:]
            boxes += torch.mul((torch.rand_like(boxes) * 2 - 1.0), diff) * box_noise_scale
            boxes = boxes.clamp(min=0.0, max=1.0)
        return boxes

    def generate_query_masks(self, max_gt_num_per_image, device):
        # 计算query数量
        noised_query_nums = max_gt_num_per_image * self.denoising_groups # 去噪query数量
        tgt_size = noised_query_nums + self.num_queries # 总query数量 = 去噪 + 匹配
        # 初始化attention mask为False（表示可以相互attend）
        attn_mask = torch.zeros(tgt_size, tgt_size, device=device, dtype=torch.bool)
        # match query cannot see the reconstruct
        # 匹配query不能看到去噪query
        # noised_query_nums: 之后的是匹配query
        # noised_query_nums 之前的是去噪query
        attn_mask[noised_query_nums:, :noised_query_nums] = True
        # 每组去噪query之间的限制
        for i in range(self.denoising_groups):
            start_col = start_row = max_gt_num_per_image * i
            end_col = end_row = max_gt_num_per_image * (i + 1)
            assert noised_query_nums >= end_col and start_col >= 0, "check attn_mask"
            # 当前组的query不能看到其他组的query
            attn_mask[start_row:end_row, :start_col] = True
            attn_mask[start_row:end_row, end_col:noised_query_nums] = True
        return attn_mask

    def forward(self, gt_labels_list, gt_boxes_list):
        """

        :param gt_labels_list: Ground truth bounding boxes per image
            with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)`
        :param gt_boxes_list: Classification labels per image in shape ``(num_gt, )``
        :return: Noised label queries, box queries, attention mask and denoising metas.
        """

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # e.g. tensor([0, 1, 2, 2, 3, 4]) -> tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]) if group = 2.
        gt_labels = gt_labels.repeat(self.denoising_groups, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # Add noise on labels and boxes
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_prob, self.num_classes)
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)
        query_num = label_embedding.shape[0]

        # add indicator to label encoding if with_indicator == True
        if self.with_indicator:
            label_embedding = torch.cat([label_embedding, torch.ones([query_num, 1], device=device)], 1)

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # the total denoising queries is depended on denoising groups and max number of instances.
        noised_query_nums = max_gt_num_per_image * self.denoising_groups

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image).long())

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([
                valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups)
            ]).long()
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image,
        )


class GenerateCDNQueries(GenerateDNQueries):
    def __init__(
        self,
        num_queries: int = 300,
        num_classes: int = 80,
        label_embed_dim: int = 256,
        denoising_nums: int = 100,
        label_noise_prob: float = 0.5,
        box_noise_scale: float = 1.0,
    ):
        super().__init__(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=label_embed_dim,
            label_noise_prob=label_noise_prob,
            box_noise_scale=box_noise_scale,
            denoising_groups=1,
        )

        self.denoising_nums = denoising_nums
        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def apply_box_noise(self, boxes: torch.Tensor, box_noise_scale: float = 0.4):
        """

        :param boxes: Bounding boxes in format ``(x_c, y_c, w, h)`` with shape ``(num_boxes, 4)``
        :param box_noise_scale: Scaling factor for box noising, defaults to 0.4
        :return: Noised boxes
        """
        # 1. 计算索引
        # 假设boxes长度为12，denoising_groups=2，则：
        num_boxes = len(boxes) // self.denoising_groups // 2
        # boxes被分为2组，每组有正负两部分，每部分3个框
        # 2. 生成positive索引
        positive_idx = torch.arange(num_boxes, dtype=torch.long, device=boxes.device)
        positive_idx = positive_idx.unsqueeze(0).repeat(self.denoising_groups, 1)
        # [[0,1,2],
        #  [0,1,2]]
        # 3. 为每组添加偏移
        positive_idx += (
            torch.arange(self.denoising_groups, dtype=torch.long, device=boxes.device).unsqueeze(1) *
            num_boxes * 2
        )
        # [[0,0,0],
        #  [6,6,6]]  # 第二组偏移6（3*2）
        # [[0,1,2],
        #  [6,7,8]]

        positive_idx = positive_idx.flatten()  # [0,1,2,6,7,8]
        negative_idx = positive_idx + num_boxes # [3,4,5,9,10,11]

        # 4. 添加噪声
        if box_noise_scale > 0:
            # 计算box的半宽和半高
            diff = torch.zeros_like(boxes)
            diff[:, :2] = boxes[:, 2:] / 2 # 中心点可移动范围为宽高的一半
            diff[:, 2:] = boxes[:, 2:] / 2 # 宽高可移动范围为自身的一半
            # 生成随机符号 (-1 或 1)
            rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            # 生成随机扰动
            rand_part = torch.rand_like(boxes) # [0,1]范围的随机值
            rand_part[negative_idx] += 1.0 # 负样本增加更大的噪声 [1,2]范围
            rand_part *= rand_sign # 乘以随机符号(-1或1)，使扰动可以是正向或负向
            # 转换为xyxy格式并添加噪声
            xyxy_boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            xyxy_boxes += torch.mul(rand_part, diff) * box_noise_scale
            # - rand_part: 随机扰动量，方向由rand_sign决定
            # - diff: 每个维度允许的最大变化范围
            # - box_noise_scale: 整体噪声的缩放因子
            xyxy_boxes = xyxy_boxes.clamp(min=0.0, max=1.0) # 限制在[0,1]范围
            boxes = box_ops._box_xyxy_to_cxcywh(xyxy_boxes)  # 转回cxcywh格式

        return boxes

    def forward(self, gt_labels_list, gt_boxes_list):
        """

        :param gt_labels_list: Classification labels per image in shape ``(num_gt, )``
        :param gt_boxes_list: Ground truth bounding boxes per image
            with normalized coordinates in format ``(x, y, w, h)`` in shape ``(num_gts, 4)`
        :return: Noised label queries, box queries, attention mask and denoising metas.
        """
        # the number of ground truth per image in one batch
        # e.g. [tensor([0, 1]), tensor([2, 3, 4])] -> gt_nums_per_image: [2, 3]
        # means there are 2 instances in the first image and 3 instances in the second image
        gt_nums_per_image = [x.numel() for x in gt_labels_list]

        # calculate the max number of ground truth in one image inside the batch.
        # e.g. gt_nums_per_image = [2, 3] which means
        # the first image has 2 instances and the second image has 3 instances
        # then the max_gt_num_per_image should be 3.
        max_gt_num_per_image = max(gt_nums_per_image)

        # get denoising_groups, which is 1 for empty ground truth
        # 自适应计算去噪组数的策略
        # 当图像中目标较少时，使用更多的去噪组
        # 当图像中目标较多时，减少去噪组数量
        denoising_groups = self.denoising_nums * max_gt_num_per_image // max(max_gt_num_per_image**2, 1)
        self.denoising_groups = max(denoising_groups, 1)

        # concat ground truth labels and boxes in one batch
        # e.g. [tensor([0, 1, 2]), tensor([2, 3, 4])] -> tensor([0, 1, 2, 2, 3, 4])
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # For efficient denoising, repeat the original ground truth labels and boxes to
        # create more training denoising samples.
        # each group has positive and negative. e.g. if group = 2, tensor([0, 1, 2, 2, 3, 4]) ->
        # tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4]).
        gt_labels = gt_labels.repeat(self.denoising_groups * 2, 1).flatten()
        gt_boxes = gt_boxes.repeat(self.denoising_groups * 2, 1)

        # set the device as "gt_labels"
        device = gt_labels.device
        assert len(gt_labels_list) == len(gt_boxes_list)

        batch_size = len(gt_labels_list)

        # Add noise on labels and boxes
        # gt_labels和gt_boxes是从多个图像concat来的
        # N = 所有图像中GT的总数
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_prob * 0.5, self.num_classes) # [N]
        noised_boxes = self.apply_box_noise(gt_boxes, self.box_noise_scale) # [N, 4]
        noised_boxes = inverse_sigmoid(noised_boxes) # [N, 4]

        # encoding labels
        label_embedding = self.label_encoder(noised_labels)

        # the total denoising queries is depended on denoising groups and max number of instances.
        # 少于max_gt_num_per_image的部分会被padding
        noised_query_nums = max_gt_num_per_image * self.denoising_groups * 2

        # initialize the generated noised queries to zero.
        # And the zero initialized queries will be assigned with noised embeddings later.
        noised_label_queries = torch.zeros(batch_size, noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(batch_size, noised_query_nums, 4, device=device)

        # batch index per image: [0, 1, 2, 3] for batch_size == 4
        batch_idx = torch.arange(0, batch_size)

        # e.g. gt_nums_per_image = [2, 3]
        # batch_idx = [0, 1]
        # then the "batch_idx_per_instance" equals to [0, 0, 1, 1, 1]
        # which indicates which image the instance (gt box) belongs to.
        # cuz the instances has been flattened before.
        batch_idx_per_instance = torch.repeat_interleave(
            batch_idx, torch.tensor(gt_nums_per_image, dtype=torch.long)
        )

        # indicate which image the noised labels belong to. For example:
        # noised label: tensor([0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4, 0, 1, 2, 2, 3, 4， 0, 1, 2, 2, 3, 4])
        # batch_idx_per_group: tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        # which means the first label "tensor([0])"" belongs to "image_0".
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups * 2, 1).flatten()

        # Cuz there might be different numbers of ground truth in each image of the same batch.
        # So there might be some padding part in noising queries.
        # Here we calculate the indexes for the valid queries and
        # fill them with the noised embeddings.
        # And leave the padding part to zeros.
        # 计算每个图像中实际的GT索引
        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([
                valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups * 2)
            ]).long()
        # 填充有效位置，其他位置保持为0
        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # generate attention masks for transformer layers
        attn_mask = self.generate_query_masks(2 * max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image * 2,
        )



# 存在的问题：

    # 问题：
    # - 噪声尺度是固定的(box_noise_scale=0.4)
    # - 没有考虑不同场景下噪声需求的差异
    # - 对所有类别使用相同的噪声策略
# 2. 简单的正负样本生成
# 正样本：小噪声
# 负样本：大噪声
# 问题：
# - 没有考虑样本难度
# - 没有考虑目标的尺度差异
# - 没有考虑目标的重叠情况


# 1. 缩放太激进：
# 目标数量翻倍，组数会减少4倍
# 可能导致目标多的图像去噪训练不足
# 2. 没有考虑目标间的关系：
# 目标多时反而可能需要更多的去噪模式
# 目标间的遮挡、重叠等复杂情况需要更多样的训练
# 3. 固定的公式：
# 没有考虑实际的训练难度
# 没有考虑不同类别的特点
# ------------------------------------------------------------
# 可能需要改进对比学习的训练方式和loss function计算
# 1.调整噪声，使用自适应的噪声，对于不同的类别和训练难度
# improved version

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.utils import box_ops
from mmdet.models.utils.misc import inverse_sigmoid

class StructuredCDNQueries(nn.Module):
    def __init__(
        self,
        num_classes=80,
        num_queries=300,
        denoising_nums=100,
        label_noise_prob=0.5,
        box_noise_scale=0.4,
        label_embed_dim=256,
        alpha=0.7
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.denoising_nums = denoising_nums
        self.label_noise_prob = label_noise_prob
        self.box_noise_scale = box_noise_scale
        self.label_embed_dim = label_embed_dim
        self.alpha = alpha

        self.label_encoder = nn.Embedding(num_classes, label_embed_dim)

    def get_spatial_relations(self, boxes):
        """计算boxes间的空间关系"""
        xyxy_boxes = box_ops._box_cxcywh_to_xyxy(boxes)
        ious = box_ops.box_iou(xyxy_boxes, xyxy_boxes)
        centers = boxes[:, :2]
        distances = torch.cdist(centers, centers)
        spatial_relation = ious + 1.0 / (distances + 1e-6)
        return spatial_relation

    def apply_structured_noise(self, boxes, box_noise_scale, is_positive):
        """
        Args:
            boxes: shape [N, 4] (cxcywh格式)
        """
        # diff shape: [N, 4]
        diff = torch.zeros_like(boxes)  # [N, 4]
        diff[:, :2] = boxes[:, 2:] / 2  # 中心点范围
        diff[:, 2:] = boxes[:, 2:]      # 尺度范围

        # rand_part shape: [N, 4]
        rand_part = torch.rand_like(boxes)  # [N, 4]
        if not is_positive:
            rand_part += 1.0
        rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part *= rand_sign  # [N, 4]

        # base_noise shape: [N, 4]
        base_noise = torch.mul(rand_part, diff) * box_noise_scale  # [N, 4]

        # spatial_relation shape: [N, N]
        spatial_relation = self.get_spatial_relations(boxes)  # [N, N]
        spatial_weights = F.softmax(spatial_relation, dim=1)  # [N, N]

        # structured_noise shape: [N, 4]
        # [N, N] @ [N, 4] -> [N, 4]
        structured_noise = torch.matmul(spatial_weights, base_noise)  # [N, 4]
        # 6. 混合噪声
        final_noise = self.alpha * structured_noise + (1-self.alpha) * base_noise

        # 7. 转换坐标系并应用噪声
        xyxy_boxes = box_ops._box_cxcywh_to_xyxy(boxes)
        noised_xyxy = xyxy_boxes + final_noise

        # 8. 转回cxcywh格式
        noised_boxes = box_ops._box_xyxy_to_cxcywh(noised_xyxy)
        noised_boxes = noised_boxes.clamp(min=0, max=1)

        # # 7. 应用噪声(TBD)
        # noised_boxes = boxes + final_noise
        # noised_boxes = noised_boxes.clamp(min=0, max=1)

        return noised_boxes

    # def apply_structured_noise(self, boxes, box_noise_scale, is_positive):
    #     """应用结构化噪声"""
    #     if len(boxes) == 0:
    #         return boxes

    #     # 1. 计算空间关系
    #     spatial_relation = self.get_spatial_relations(boxes)

    #     # 2. 生成基础噪声
    #     diff = torch.zeros_like(boxes)
    #     diff[:, :2] = boxes[:, 2:] / 2
    #     diff[:, 2:] = boxes[:, 2:]

    #     # 3. 生成随机噪声
    #     rand_part = torch.rand_like(boxes)
    #     if not is_positive:
    #         rand_part += 1.0  # 负样本增加噪声范围

    #     rand_sign = torch.randint_like(boxes, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
    #     rand_part *= rand_sign

    #     # 4. 生成基础噪声
    #     base_noise = torch.mul(rand_part, diff) * box_noise_scale

    #     # 5. 应用空间约束
    #     spatial_weights = F.softmax(spatial_relation, dim=1)
    #     structured_noise = torch.matmul(spatial_weights, base_noise)

    #     # 6. 混合噪声
    #     final_noise = self.alpha * structured_noise + (1-self.alpha) * base_noise

    #     # 7. 应用噪声
    #     noised_boxes = boxes + final_noise
    #     noised_boxes = noised_boxes.clamp(min=0, max=1)

    #     return noised_boxes

    def apply_label_noise(self, labels, noise_prob, num_classes):
        if noise_prob > 0:
            noise_labels = torch.randint_like(labels, 0, num_classes)
            noise_mask = torch.rand_like(labels.float()) < noise_prob
            labels = torch.where(noise_mask, noise_labels, labels)
        return labels

    def generate_attention_mask(self, max_gt_num_per_image, device):
        mask = torch.zeros(max_gt_num_per_image, max_gt_num_per_image, device=device)
        mask = torch.where(mask == 1, float('-inf'), mask)
        return mask

    def forward(self, gt_labels_list, gt_boxes_list):
        # 1. 获取每张图像的GT数量
        gt_nums_per_image = [x.numel() for x in gt_labels_list]
        max_gt_num_per_image = max(gt_nums_per_image)

        # 2. 计算去噪组数
        denoising_groups = self.denoising_nums * max_gt_num_per_image // max(max_gt_num_per_image**2, 1)
        self.denoising_groups = max(denoising_groups, 1)

        # 3. 合并batch中的GT
        gt_labels = torch.cat(gt_labels_list)
        gt_boxes = torch.cat(gt_boxes_list)

        # 4. 重复GT用于多组去噪
        gt_labels = gt_labels.repeat(self.denoising_groups * 2)
        gt_boxes = gt_boxes.repeat(self.denoising_groups * 2, 1)

        # 5. 应用噪声
        device = gt_labels.device
        noised_labels = self.apply_label_noise(gt_labels, self.label_noise_prob * 0.5, self.num_classes)

        # 计算总的GT数量
        total_gts = sum(gt_nums_per_image)

        # 按图像和组应用结构化噪声
        start_idx = 0
        noised_boxes_list = []

        for num_gt in gt_nums_per_image:
            for group_idx in range(self.denoising_groups):
                # 计算当前组的索引
                group_offset = group_idx * total_gts * 2

                # 获取当前图像的正样本和负样本boxes
                curr_pos_boxes = gt_boxes[group_offset + start_idx:group_offset + start_idx + num_gt]
                curr_neg_boxes = gt_boxes[group_offset + total_gts + start_idx:group_offset + total_gts + start_idx + num_gt]

                # 应用结构化噪声
                noised_pos = self.apply_structured_noise(curr_pos_boxes, self.box_noise_scale, is_positive=True)
                noised_neg = self.apply_structured_noise(curr_neg_boxes, self.box_noise_scale, is_positive=False)

                noised_boxes_list.extend([noised_pos, noised_neg])

            start_idx += num_gt

        noised_boxes = torch.cat(noised_boxes_list)
        noised_boxes = inverse_sigmoid(noised_boxes)

        # 6. 编码标签
        label_embedding = self.label_encoder(noised_labels)

        # 7. 初始化查询
        noised_query_nums = max_gt_num_per_image * self.denoising_groups * 2
        noised_label_queries = torch.zeros(len(gt_labels_list), noised_query_nums, self.label_embed_dim, device=device)
        noised_box_queries = torch.zeros(len(gt_labels_list), noised_query_nums, 4, device=device)

        # 8. 填充有效查询
        batch_idx = torch.arange(len(gt_labels_list))
        batch_idx_per_instance = torch.repeat_interleave(batch_idx, torch.tensor(gt_nums_per_image, dtype=torch.long))
        batch_idx_per_group = batch_idx_per_instance.repeat(self.denoising_groups * 2, 1).flatten()

        if len(gt_nums_per_image):
            valid_index_per_group = torch.cat([torch.arange(num) for num in gt_nums_per_image])
            valid_index_per_group = torch.cat([
                valid_index_per_group + max_gt_num_per_image * i for i in range(self.denoising_groups * 2)
            ]).long()

        if len(batch_idx_per_group):
            noised_label_queries[(batch_idx_per_group, valid_index_per_group)] = label_embedding
            noised_box_queries[(batch_idx_per_group, valid_index_per_group)] = noised_boxes

        # 9. 生成注意力掩码
        attn_mask = self.generate_attention_mask(2 * max_gt_num_per_image, device)

        return (
            noised_label_queries,
            noised_box_queries,
            attn_mask,
            self.denoising_groups,
            max_gt_num_per_image * 2,
        )