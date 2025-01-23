import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops.boxes import _box_cxcywh_to_xyxy, generalized_box_iou, box_iou


class HungarianMatcher(nn.Module):
    """This class implements the Hungarian matching algorithm for bipartite graphs. It matches predicted bounding
    boxes to ground truth boxes based on the minimum cost assignment. The cost is computed as a weighted sum of
    classification, bounding box, and generalized intersection over union (IoU) costs. The focal loss is used to
    weigh the classification cost. The HungarianMatcher class can be used in single or mixed assignment modes.
    The mixed assignment modes is introduced in `Align-DETR <https://arxiv.org/abs/2304.07527>`_.

    :param cost_class: The weight of the classification cost, defaults to 1
    :param cost_bbox: The weight of the bounding box cost, defaults to 1
    :param cost_giou: The weight of the generalized IoU cost, defaults to 1
    :param focal_alpha: The alpha parameter of the focal loss, defaults to 0.25
    :param focal_gamma: The gamma parameter of the focal loss, defaults to 2.0
    :param mixed_match: If True, mixed assignment is used, defaults to False
    """
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mixed_match: bool = False,
    ):
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mixed_match = mixed_match

    def calculate_class_cost(self, pred_logits, gt_labels, **kwargs):
        out_prob = pred_logits.sigmoid()

        # Compute the classification cost.
        neg_cost_class = -(1 - self.focal_alpha) * out_prob**self.focal_gamma * (1 - out_prob + 1e-6).log()
        pos_cost_class = -self.focal_alpha * (1 - out_prob)**self.focal_gamma * (out_prob + 1e-6).log()
        cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]
        # focal loss: 容易样本（预测很准的），权重较小; 困难样本（预测不准的），权重较大
        # cost_class: (num_queries, num_gt_boxes 预测越准确，代价越小
        return cost_class

    def calculate_bbox_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
        return cost_bbox

    def calculate_giou_cost(self, pred_boxes, gt_boxes, **kwargs):
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(_box_cxcywh_to_xyxy(pred_boxes), _box_cxcywh_to_xyxy(gt_boxes))
        return cost_giou

    @torch.no_grad()
    def calculate_cost(self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor):
        # Calculate class, bbox and giou cost
        # 由于使用的是Focal Loss:
        # - log(p)的范围是[0, ∞)
        # - focal weight (1-p)^γ 的范围是[0, 1]
        # - alpha的范围是[0, 1]，通常是0.25
        # 实际范围大约在[0, 7]之间，但理论上没有上限
        cost_class = self.calculate_class_cost(pred_logits, gt_labels)
        # 使用L1距离（曼哈顿距离）计算
        # boxes是归一化的坐标(cx, cy, w, h)，范围都在[0, 1]
        # 因此L1距离的范围是[0, 4]
        # - 每个坐标差的最大值是1
        # - 4个坐标，所以最大总和是4
        cost_bbox = self.calculate_bbox_cost(pred_boxes, gt_boxes)
        # GIoU的范围是[-1, 1]
        # 由于代码中使用的是-GIoU
        # 所以cost_giou的范围是[-1, 1]
        cost_giou = self.calculate_giou_cost(pred_boxes, gt_boxes)

        # Final cost matrix
        c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        return c

    @torch.no_grad()
    def forward(
        self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor, gt_copy: int = 1
    ):
        c = self.calculate_cost(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # single assignment
        if not self.mixed_match:
            indices = linear_sum_assignment(c.cpu())
            # indices[0] = [1, 2]     # queries的索引
            # indices[1] = [0, 1]     # gt_boxes的索引
            return torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

        # mixed assignment, used in AlignDETR
        gt_size = c.size(-1) # ground truth的数量
        num_queries = len(c)
        # 每个target最多被匹配的次数，不超过queries数量的一半除以target数量
        gt_copy = min(int(num_queries * 0.5 / gt_size), gt_copy) if gt_size > 0 else gt_copy
        # 水平方向重复gt_copy次
        # 例如，原cost矩阵：
        # [[0.1, 0.2],
        #  [0.3, 0.4],
        #  [0.5, 0.6]]
        # gt_copy=2时，扩展后：
        # [[0.1, 0.2, 0.1, 0.2],
        #  [0.3, 0.4, 0.3, 0.4],
        #  [0.5, 0.6, 0.5, 0.6]]
        src_ind, tgt_ind = linear_sum_assignment(c.cpu().repeat(1, gt_copy))
        # 还原真实的target索引
        tgt_ind = tgt_ind % gt_size
        # 对target索引排序，并相应调整source索引
        tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
        src_ind = torch.as_tensor(src_ind, dtype=torch.int64)[ind].view(-1)
        return src_ind, tgt_ind


class StableHungarianMatcher(HungarianMatcher):
    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mixed_match: bool = False,
        debug: bool = False,
    ):
        super().__init__(
            cost_class, cost_bbox, cost_giou, focal_alpha, focal_gamma, mixed_match)
        self.debug = debug

    @torch.no_grad()
    def forward(
        self,
        pred_boxes: Tensor,
        pred_logits: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        gt_copy: int = 1, # lower bound,
        k: int = 3, # upper bound topk
    ):
        self.k = k
        c = self.calculate_cost(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # single assignment
        if not self.mixed_match:
            indices = linear_sum_assignment(c.cpu())
            # indices[0] = [1, 2]     # queries的索引
            # indices[1] = [0, 1]     # gt_boxes的索引
            return torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

        # mixed assignment, used in AlignDETR
        gt_size = c.size(-1) # ground truth的数量
        num_queries = len(c)
        if gt_size > 0 and int(num_queries * 0.5 / gt_size) <= gt_copy:
            gt_copy = int(num_queries * 0.5 / gt_size)
            src_ind, tgt_ind = linear_sum_assignment(c.cpu().repeat(1, gt_copy))
            # 还原真实的target索引
            tgt_ind = tgt_ind % gt_size
            # 对target索引排序，并相应调整source索引
            tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
            src_ind = torch.as_tensor(src_ind, dtype=torch.int64)[ind].view(-1)
            return src_ind, tgt_ind

        dynamic_copies, top_k_scores, score_sums = self._get_dynamic_copies(pred_boxes, pred_logits, gt_boxes, gt_labels)

        # 5. 构建扩展的cost matrix
        expanded_costs = []
        start_indices = [0]

        for gt_idx in range(len(gt_boxes)):
            n_copies = dynamic_copies[gt_idx].item()
            expanded_costs.append(c[:, gt_idx:gt_idx+1].repeat(1, n_copies))
            start_indices.append(start_indices[-1] + n_copies)

        expanded_costs = torch.cat(expanded_costs, dim=1)
        src_ind, tgt_ind = linear_sum_assignment(expanded_costs.cpu())

        original_gt_indices = torch.zeros_like(torch.tensor(tgt_ind))
        for gt_idx in range(len(gt_boxes)):
            start_idx = start_indices[gt_idx]
            end_idx = start_indices[gt_idx + 1]
            mask = (tgt_ind >= start_idx) & (tgt_ind < end_idx)
            original_gt_indices[mask] = gt_idx

        # 调试信息
        if self.debug:
            print("\nMatching info:")
            print(f"Number of GTs: {len(gt_boxes)}")
            print("GT details:")
            for gt_idx in range(len(gt_boxes)):
                print(f"GT {gt_idx}:")
                print(f"  - Top-k scores: {top_k_scores[:, gt_idx].tolist()}")
                print(f"  - Score sum: {score_sums[gt_idx]:.3f}")
                print(f"  - Assigned copies: {dynamic_copies[gt_idx].item()}")
                print(f"  - Actual matches: {(original_gt_indices == gt_idx).sum()}")

        return src_ind, original_gt_indices


    def _get_dynamic_copies(self, pred_boxes, pred_logits, gt_boxes, gt_labels, gt_copy):
        # 计算每个gt的top-k匹配质量，决定copies数量
        prob = pred_logits.softmax(-1)  # [num_queries, num_classes]
        scores = prob[..., gt_labels]   # [num_queries, num_gt]
        ious = box_iou(pred_boxes, gt_boxes)[0]  # [num_queries, num_gt]

        match_scores = ious * (1 - scores)  # [num_queries, num_gt]
        # match_scores = ious * (1 - scores**2)

        # 对每个gt，选择top-k个最佳匹配的得分, upper_bound=k
        top_k_scores, _ = match_scores.topk(k=self.k, dim=0)  # [k, num_gt]
        score_sums = top_k_scores.sum(dim=0) # [num_gt]
        dynamic_copies = torch.maximum(
            int(score_sums), torch.tensor(gt_copy))

        return dynamic_copies, top_k_scores, score_sums



# class HybridStableHungarianMatcher(HungarianMatcher):
#     def calculate_class_cost(self, pred_logits, gt_labels, pred_boxes=None, gt_boxes=None):
#         out_prob = pred_logits.sigmoid()

#         # 1. 计算IoU
#         ious = box_iou(_box_cxcywh_to_xyxy(pred_boxes), _box_cxcywh_to_xyxy(gt_boxes))

#         # 2. 计算分类cost时考虑IoU
#         neg_cost_class = -(1 - self.focal_alpha) * out_prob**self.focal_gamma * (1 - out_prob + 1e-6).log()
#         pos_cost_class = -self.focal_alpha * (1 - out_prob)**self.focal_gamma * (out_prob + 1e-6).log()
#         base_cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

#         # # 使用IoU调整分类cost
#         # cost_class = base_cost_class * (1 + ious)

#         return cost_class



# +----------------+------+-------+-------+--------+-------+
# | class          | imgs | gts   | dets  | recall | ap    |
# +----------------+------+-------+-------+--------+-------+
# | person         | 2693 | 11004 | 48322 | 0.961  | 0.860 |
# | bicycle        | 149  | 316   | 2550  | 0.901  | 0.653 |
# | car            | 535  | 1932  | 28206 | 0.937  | 0.744 |
# | motorcycle     | 159  | 371   | 2402  | 0.951  | 0.766 |
# | airplane       | 97   | 143   | 1582  | 0.993  | 0.898 |
# | bus            | 189  | 285   | 2024  | 0.958  | 0.854 |
# | train          | 157  | 190   | 2262  | 0.984  | 0.902 |
# | truck          | 250  | 415   | 3953  | 0.969  | 0.599 |
# | boat           | 121  | 430   | 6300  | 0.913  | 0.602 |
# | traffic light  | 191  | 637   | 6160  | 0.879  | 0.612 |
# | fire hydrant   | 86   | 101   | 730   | 0.980  | 0.889 |
# | stop sign      | 69   | 75    | 1314  | 0.960  | 0.778 |
# | parking meter  | 37   | 60    | 797   | 0.917  | 0.681 |
# | bench          | 235  | 413   | 11318 | 0.800  | 0.434 |
# | bird           | 125  | 440   | 9777  | 0.829  | 0.658 |
# | cat            | 184  | 202   | 2164  | 0.995  | 0.921 |
# | dog            | 177  | 218   | 5505  | 0.972  | 0.854 |
# | horse          | 128  | 273   | 1832  | 0.960  | 0.853 |
# | sheep          | 65   | 361   | 2500  | 0.969  | 0.857 |
# | cow            | 87   | 380   | 2643  | 0.973  | 0.846 |
# | elephant       | 89   | 255   | 2833  | 0.996  | 0.902 |
# | bear           | 49   | 71    | 4019  | 0.986  | 0.954 |
# | zebra          | 85   | 268   | 2493  | 0.992  | 0.932 |
# | giraffe        | 101  | 232   | 1631  | 0.991  | 0.918 |
# | backpack       | 228  | 371   | 6875  | 0.884  | 0.349 |
# | umbrella       | 174  | 413   | 2348  | 0.924  | 0.690 |
# | handbag        | 292  | 540   | 6848  | 0.874  | 0.392 |
# | tie            | 145  | 254   | 2915  | 0.845  | 0.640 |
# | suitcase       | 105  | 303   | 1959  | 0.930  | 0.718 |
# | frisbee        | 84   | 115   | 868   | 0.983  | 0.934 |
# | skis           | 120  | 241   | 4472  | 0.892  | 0.588 |
# | snowboard      | 49   | 69    | 1573  | 0.913  | 0.587 |
# | sports ball    | 169  | 263   | 3761  | 0.892  | 0.758 |
# | kite           | 91   | 336   | 1954  | 0.930  | 0.740 |
# | baseball bat   | 97   | 146   | 1196  | 0.869  | 0.654 |
# | baseball glove | 100  | 148   | 1418  | 0.919  | 0.726 |
# | skateboard     | 127  | 179   | 1725  | 0.933  | 0.824 |
# | surfboard      | 149  | 269   | 5317  | 0.906  | 0.733 |
# | tennis racket  | 167  | 225   | 2167  | 0.911  | 0.818 |
# | bottle         | 379  | 1025  | 6429  | 0.927  | 0.654 |
# | wine glass     | 110  | 343   | 1227  | 0.865  | 0.655 |
# | cup            | 390  | 899   | 5192  | 0.928  | 0.669 |
# | fork           | 155  | 215   | 2053  | 0.953  | 0.657 |
# | knife          | 181  | 326   | 2897  | 0.828  | 0.459 |
# | spoon          | 153  | 253   | 3152  | 0.838  | 0.412 |
# | bowl           | 314  | 626   | 4643  | 0.947  | 0.642 |
# | banana         | 103  | 379   | 3094  | 0.878  | 0.445 |
# | apple          | 76   | 239   | 2080  | 0.843  | 0.382 |
# | sandwich       | 98   | 177   | 1138  | 0.938  | 0.601 |
# | orange         | 85   | 287   | 987   | 0.888  | 0.515 |
# | broccoli       | 71   | 316   | 4405  | 0.920  | 0.446 |
# | carrot         | 3    | 2303  | 28206 | 0.871  | 0.414 |
# | hot dog        | 0    | 345   | 5505  | 0.848  | 0.642 |
# | pizza          | 153  | 285   | 1543  | 0.951  | 0.795 |
# | donut          | 62   | 338   | 1308  | 0.957  | 0.772 |
# | cake           | 124  | 316   | 2127  | 0.932  | 0.665 |
# | chair          | 580  | 1791  | 15509 | 0.887  | 0.548 |
# | couch          | 195  | 261   | 3889  | 0.958  | 0.657 |
# | potted plant   | 172  | 343   | 7880  | 0.901  | 0.520 |
# | bed            | 149  | 163   | 3087  | 0.988  | 0.664 |
# | dining table   | 501  | 697   | 8340  | 0.894  | 0.487 |
# | toilet         | 149  | 179   | 2669  | 0.978  | 0.846 |
# | tv             | 207  | 288   | 2122  | 0.951  | 0.817 |
# | laptop         | 183  | 231   | 1163  | 0.948  | 0.809 |
# | mouse          | 88   | 106   | 484   | 0.934  | 0.839 |
# | remote         | 145  | 283   | 1806  | 0.940  | 0.633 |
# | keyboard       | 106  | 153   | 817   | 0.941  | 0.755 |
# | cell phone     | 214  | 262   | 3021  | 0.882  | 0.641 |
# | microwave      | 54   | 55    | 393   | 0.945  | 0.804 |
# | oven           | 115  | 143   | 1601  | 0.951  | 0.580 |
# | toaster        | 8    | 9     | 158   | 1.000  | 0.563 |
# | sink           | 187  | 225   | 3880  | 0.924  | 0.669 |
# | refrigerator   | 101  | 126   | 698   | 0.944  | 0.791 |
# | book           | 230  | 1161  | 8288  | 0.846  | 0.391 |
# | clock          | 204  | 267   | 3814  | 0.955  | 0.793 |
# | vase           | 137  | 277   | 2907  | 0.905  | 0.626 |
# | scissors       | 28   | 36    | 1040  | 0.778  | 0.541 |
# | teddy bear     | 0    | 262   | 4019  | 0.926  | 0.702 |
# | hair drier     | 9    | 11    | 378   | 0.727  | 0.371 |
# | toothbrush     | 34   | 57    | 1449  | 0.754  | 0.462 |
# Create dictionary from raw data
# gt_dict = {
#     'person': 11004, 'bicycle': 316, 'car': 1932, 'motorcycle': 371,
#     'airplane': 143, 'bus': 285, 'train': 190, 'truck': 415,
#     'boat': 430, 'traffic light': 637, 'fire hydrant': 101, 'stop sign': 75,
#     'parking meter': 60, 'bench': 413, 'bird': 440, 'cat': 202,
#     'dog': 218, 'horse': 273, 'sheep': 361, 'cow': 380,
#     'elephant': 255, 'bear': 71, 'zebra': 268, 'giraffe': 232,
#     'backpack': 371, 'umbrella': 413, 'handbag': 540, 'tie': 254,
#     'suitcase': 303, 'frisbee': 115, 'skis': 241, 'snowboard': 69,
#     'sports ball': 263, 'kite': 336, 'baseball bat': 146, 'baseball glove': 148,
#     'skateboard': 179, 'surfboard': 269, 'tennis racket': 225, 'bottle': 1025,
#     'wine glass': 343, 'cup': 899, 'fork': 215, 'knife': 326,
#     'spoon': 253, 'bowl': 626, 'banana': 379, 'apple': 239,
#     'sandwich': 177, 'orange': 287, 'broccoli': 316, 'carrot': 2303,
#     'hot dog': 345, 'pizza': 285, 'donut': 338, 'cake': 316,
#     'chair': 1791, 'couch': 261, 'potted plant': 343, 'bed': 163,
#     'dining table': 697, 'toilet': 179, 'tv': 288, 'laptop': 231,
#     'mouse': 106, 'remote': 283, 'keyboard': 153, 'cell phone': 262,
#     'microwave': 55, 'oven': 143, 'toaster': 9, 'sink': 225,
#     'refrigerator': 126, 'book': 1161, 'clock': 267, 'vase': 277,
#     'scissors': 36, 'teddy bear': 262, 'hair drier': 11, 'toothbrush': 57
# }
# Create dictionary mapping label IDs to ground truth counts
id_to_gt = {
    1: 11004,  # person
    2: 316,    # bicycle
    3: 1932,   # car
    4: 371,    # motorcycle
    5: 143,    # airplane
    6: 285,    # bus
    7: 190,    # train
    8: 415,    # truck
    9: 430,    # boat
    10: 637,   # traffic light
    11: 101,   # fire hydrant
    13: 75,    # stop sign
    14: 60,    # parking meter
    15: 413,   # bench
    16: 440,   # bird
    17: 202,   # cat
    18: 218,   # dog
    19: 273,   # horse
    20: 361,   # sheep
    21: 380,   # cow
    22: 255,   # elephant
    23: 71,    # bear
    24: 268,   # zebra
    25: 232,   # giraffe
    27: 371,   # backpack
    28: 413,   # umbrella
    31: 540,   # handbag
    32: 254,   # tie
    33: 303,   # suitcase
    34: 115,   # frisbee
    35: 241,   # skis
    36: 69,    # snowboard
    37: 263,   # sports ball
    38: 336,   # kite
    39: 146,   # baseball bat
    40: 148,   # baseball glove
    41: 179,   # skateboard
    42: 269,   # surfboard
    43: 225,   # tennis racket
    44: 1025,  # bottle
    46: 343,   # wine glass
    47: 899,   # cup
    48: 215,   # fork
    49: 326,   # knife
    50: 253,   # spoon
    51: 626,   # bowl
    52: 379,   # banana
    53: 239,   # apple
    54: 177,   # sandwich
    55: 287,   # orange
    56: 316,   # broccoli
    57: 2303,  # carrot
    58: 345,   # hot dog
    59: 285,   # pizza
    60: 338,   # donut
    61: 316,   # cake
    62: 1791,  # chair
    63: 261,   # couch
    64: 343,   # potted plant
    65: 163,   # bed
    67: 697,   # dining table
    70: 179,   # toilet
    72: 288,   # tv
    73: 231,   # laptop
    74: 106,   # mouse
    75: 283,   # remote
    76: 153,   # keyboard
    77: 262,   # cell phone
    78: 55,    # microwave
    79: 143,   # oven
    80: 9,     # toaster
    81: 225,   # sink
    82: 126,   # refrigerator
    84: 1161,  # book
    85: 267,   # clock
    86: 277,   # vase
    87: 36,    # scissors
    88: 262,   # teddy bear
    89: 11,    # hair drier
    90: 57,    # toothbrush
}
# Sort dictionary by values in descending order
sorted_dict = dict(sorted(id_to_gt.items(), key=lambda x: x[1], reverse=True))

# class SpeaQHungarianMatcher(nn.Module):
#     def __init__(
#         self,
#         cost_class: float = 1,
#         cost_bbox: float = 1,
#         cost_giou: float = 1,
#         focal_alpha: float = 0.25,
#         focal_gamma: float = 2.0,
#         mixed_match: bool = False,
#         num_groups: int = 5,
#         num_classes: int = 91,
#         num_mul_so_queries: int = 900, # multiple specialist queries
#     ):
#         super().__init__()

#         self.cost_class = cost_class
#         self.cost_bbox = cost_bbox
#         self.cost_giou = cost_giou
#         assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma
#         self.mixed_match = mixed_match

#         self.num_groups = num_groups
#         self.num_classes = num_classes
#         self.num_mul_so_queries = num_mul_so_queries
#         # there are only 80 classes in COCO but label id is from 1 to 90 some of them are not used
#         self.class_freq = torch.tensor(list(sorted_dict.values()))
#         self.class_order = torch.tensor(list(sorted_dict.keys()))
#         # 2种分组:
#         # 1. 将classes按照频率进行分组
#         # 2. 将queries按照相应比例进行分组
#         # assume num_groups = 3, num_classes = 5
#         # 返回一个列表，包含每个组应该包含的关系数量。比如 [1, 2, 2] 表示第一组包含1个class，第二组包含2个class，第三组包含2个class
#         self.size_of_groups = self.get_group_list_by_n_groups(self.num_groups)
#         self.grouping()
#         print(f'query assignment: {self.freq_list}')

#     def get_group_list_by_n_groups(self, n_groups):
#         class_freq_np = self.class_freq.numpy()
#         total_list = []
#         last_checked_index = 0
#         current_idx = 0
#         size_of_whole_groups = 0

#         for i in range(n_groups - 1):
#             sum_of_this_group = 0
#             size_of_this_group = 0
#             remaining_list = class_freq_np[last_checked_index:]
#             remaining_half_cnt = remaining_list.sum() // 2

#             while (current_idx < len(class_freq_np) and
#                 sum_of_this_group + class_freq_np[current_idx] < remaining_half_cnt):
#                 sum_of_this_group += class_freq_np[current_idx]
#                 size_of_this_group += 1
#                 size_of_whole_groups += 1
#                 current_idx += 1

#             if size_of_this_group == 0:  # 防止出现空组
#                 size_of_this_group = 1
#                 size_of_whole_groups += 1
#                 current_idx += 1

#             total_list.append(size_of_this_group)
#             last_checked_index = current_idx

#         # 确保最后一组至少有一个元素
#         last_group_size = max(1, len(self.class_freq) - size_of_whole_groups)
#         total_list.append(last_group_size)

#         print(f'total_list: {total_list}, size_of_groups: {len(total_list)}, sum of total_list: {sum(total_list)}')

#         return total_list

#     def fill_list(self, num, n):
#         quotient, remainder = divmod(num, n)
#         lst = [quotient] * n
#         for i in range(remainder):
#             lst[-1 * (i + 1)] += 1
#         return torch.tensor(lst)

#     def grouping(self):
#         device_group = 'cuda'
#         group_tensor = -torch.ones(self.num_classes, device=device_group)
#         # 计算每个组的class频率总和
#         sum_of_each_groups = torch.as_tensor(
#             [x.sum().item() for x in torch.split(self.class_freq, self.size_of_groups)], device=device_group)
#         # 根据频率比例分配查询数量
#         n_queries_per_group = (sum_of_each_groups * self.num_mul_so_queries / sum_of_each_groups.sum()).int()
#         # 处理舍入误差，确保查询总数正确
#         n_queries_per_group += self.fill_list((self.num_mul_so_queries - n_queries_per_group.sum()).item(), len(n_queries_per_group)).to(device=device_group)
#         self.n_queries_per_group = n_queries_per_group.long()
#         assert self.num_mul_so_queries == n_queries_per_group.sum()

#         # 将class按照size_of_groups分割
#         self.class_rel_order = torch.split(self.class_order, self.size_of_groups)
#         # 为每个class分配组ID class_id -> group_id
#         for g, row in enumerate(self.class_rel_order):
#             group_tensor[row] = g
#         self.group_tensor = group_tensor # 存储每个class属于哪个组
#         self.freq_list = torch.tensor(n_queries_per_group.cpu().numpy()) # 存储每个组分 group_id -> num_queries
#         self.n_groups = len(self.freq_list) # 存储组的总数


#     def calculate_class_cost(self, pred_logits, gt_labels, **kwargs):
#         out_prob = pred_logits.sigmoid()

#         # Compute the classification cost.
#         neg_cost_class = -(1 - self.focal_alpha) * out_prob**self.focal_gamma * (1 - out_prob + 1e-6).log()
#         pos_cost_class = -self.focal_alpha * (1 - out_prob)**self.focal_gamma * (out_prob + 1e-6).log()
#         cost_class = pos_cost_class[:, gt_labels] - neg_cost_class[:, gt_labels]

#         return cost_class

#     def calculate_bbox_cost(self, pred_boxes, gt_boxes, **kwargs):
#         # Compute the L1 cost between boxes
#         cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)
#         return cost_bbox

#     def calculate_giou_cost(self, pred_boxes, gt_boxes, **kwargs):
#         # Compute the giou cost betwen boxes
#         cost_giou = -generalized_box_iou(_box_cxcywh_to_xyxy(pred_boxes), _box_cxcywh_to_xyxy(gt_boxes))
#         return cost_giou

#     @torch.no_grad()
#     def calculate_cost(self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor):
#         # Calculate class, bbox and giou cost
#         cost_class = self.calculate_class_cost(pred_logits, gt_labels)
#         cost_bbox = self.calculate_bbox_cost(pred_boxes, gt_boxes)
#         cost_giou = self.calculate_giou_cost(pred_boxes, gt_boxes)

#         # Final cost matrix
#         c = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

#         # 获取每个gt_label对应的组ID
#         gt_groups = self.group_tensor[gt_labels].long()  # [num_gt_boxes]

#         # 计算每个组的queries的起始索引
#         start_indices = torch.zeros_like(self.freq_list, device=c.device)
#         start_indices[1:] = torch.cumsum(torch.tensor(self.freq_list[:-1], device=c.device), dim=0)

#         # 创建一个mask，初始化为False (或者1e6的cost)
#         mask = torch.ones_like(c) * 1e6
#         # 对每个gt box
#         for gt_idx, group_id in enumerate(gt_groups):
#             # 获取该组的queries的起始和结束索引
#             start_idx = start_indices[group_id]
#             end_idx = start_idx + self.freq_list[group_id]
#             # 将对应范围内的cost保持原值，其他设为一个大数
#             mask[start_idx:end_idx, gt_idx] = 0

#         # 应用mask torch.where(condition, x, y)
#         c = torch.where(mask == 0, c, mask)
#         # TODO: 将cost矩阵按照类的group进行specialization
#         return c

#     @torch.no_grad()
#     def forward(
#         self, pred_boxes: Tensor, pred_logits: Tensor, gt_boxes: Tensor, gt_labels: Tensor, gt_copy: int = 1
#     ):
#         # c: (num_queries, num_gt_boxes)
#         c = self.calculate_cost(pred_boxes, pred_logits, gt_boxes, gt_labels)

#         # c = c.view(self.num_groups, -1)

#         # single assignment
#         if not self.mixed_match:
#             indices = linear_sum_assignment(c.cpu())
#             return torch.as_tensor(indices[0]), torch.as_tensor(indices[1])

#         # mixed assignment, used in AlignDETR
#         gt_size = c.size(-1)
#         num_queries = len(c)
#         gt_copy = min(int(num_queries * 0.5 / gt_size), gt_copy) if gt_size > 0 else gt_copy
#         src_ind, tgt_ind = linear_sum_assignment(c.cpu().repeat(1, gt_copy))
#         tgt_ind = tgt_ind % gt_size
#         tgt_ind, ind = torch.as_tensor(tgt_ind, dtype=torch.int64).sort()
#         src_ind = torch.as_tensor(src_ind, dtype=torch.int64)[ind].view(-1)
#         return src_ind, tgt_ind


# class StableHungarianMatcher(HungarianMatcher):
#     def __init__(
#         self,
#         cost_class: float = 1.0,
#         cost_bbox: float = 1.0,
#         cost_giou: float = 1.0,
#         focal_alpha: float = 0.25,
#         focal_gamma: float = 2.0,
#         stability_weight: float = 0.2,
#     ):
#         super().__init__(
#             cost_class=cost_class,
#             cost_bbox=cost_bbox,
#             cost_giou=cost_giou,
#             focal_alpha=focal_alpha,
#             focal_gamma=focal_gamma,
#         )
#         self.stability_weight = stability_weight
#         self.layer_matches = {}  # 存储每一层的匹配

    # def forward(self, pred_boxes, pred_logits, gt_boxes, gt_labels,
    #             is_encoder=False, batch_idx=None, layer_idx=None):
    #     # 使用父类的calculate_cost方法
    #     C = self.calculate_cost(pred_logits, pred_boxes, gt_labels, gt_boxes)

#         if self.training and not is_encoder and batch_idx is not None:
#             if layer_idx is not None:  # 辅助层
#                 # 获取下一层的匹配结果
#                 next_layer_idx = layer_idx + 1
#                 if batch_idx in self.layer_matches and next_layer_idx in self.layer_matches[batch_idx]:
#                     prev_matches = self.layer_matches[batch_idx][next_layer_idx]
#                     stability_cost = self.calculate_stability_cost(
#                         C.shape,
#                         prev_matches,
#                         C.device
#                     )
#                     C = C + self.stability_weight * stability_cost

#         indices = linear_sum_assignment(C.cpu())
#         indices = (
#             torch.as_tensor(indices[0], dtype=torch.int64),
#             torch.as_tensor(indices[1], dtype=torch.int64)
#         )

#         # 保存当前层的匹配结果
#         if self.training and not is_encoder and batch_idx is not None:
#             if batch_idx not in self.layer_matches:
#                 self.layer_matches[batch_idx] = {}
#             current_layer_idx = layer_idx if layer_idx is not None else 5 # hardcode idx 5 表示最后一层
#             self.layer_matches[batch_idx][current_layer_idx] = indices

#         return indices

#     def calculate_stability_cost(self, shape, prev_matches, device):
#         num_queries, num_targets = shape
#         stability_cost = torch.ones((num_queries, num_targets), device=device)
#         prev_q, prev_t = prev_matches
#         stability_cost[prev_q, prev_t] = 0.0
#         return stability_cost

    # def reset_matches(self):
    #     self.layer_matches.clear()

# 5. 需要注意的点：
# stability_weight的选择很重要
# 太大：可能强制不合理的匹配
# 太小：可能没有效果
# 是否要在所有层间保持稳定性
# 可能早期层允许更多变化
# 后期层要求更稳定

# 可能的变体
# 动态调整稳定性权重
# stability_weight = self.stability_weight * (1 - math.exp(-current_epoch/10))

# # 基于匹配质量的稳定性权重
# quality_based_weight = self.stability_weight * match_quality_score

# # 渐进式稳定性
# if current_epoch > stability_start_epoch:
#     # 添加稳定性约束
