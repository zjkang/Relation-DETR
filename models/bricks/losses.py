from torch.nn import functional as F


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    target_score = targets.to(inputs.dtype)
    weight = (1 - alpha) * prob**gamma * (1 - targets) + targets * alpha * (1 - prob)**gamma
    # according to original implementation, sigmoid_focal_loss keep gradient on weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    loss = loss * weight
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def vari_sigmoid_focal_loss(inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid().detach()  # pytorch version of RT-DETR has detach while paddle version not
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def ia_bce_loss(inputs, targets, gt_score, num_boxes, k: float = 0.25, alpha: float = 0, gamma: float = 2):
    # inputs: [num_queries, num_classes] 或 [batch_size, num_queries, num_classes]

    # targets: 通常是one-hot标签，[num_queries, num_classes] 或 [batch_size, num_queries, num_classes]
    # 例如：
    # batch_size = 1
    # num_queries = 100
    # num_targets = 2  # 一张图片有2个目标
    # num_classes = 80
    # # 假设：
    # # target_0 是类别7
    # # target_1 是类别15
    # # query_5 匹配到了 target_0
    # # query_23 匹配到了 target_1
    # # 其他query都是负样本
    # targets[5] = [0,0,0,0,0,0,0,1,0,...,0]  # 第7类为1
    # targets[23] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,...,0]  # 第15类为1
    # # 其他query的targets全为0

    # gt_score: [num_queries] 通常表示预测框与对应真实框（ground truth）的IoU值。这个分数反映了定位的质量
    # gt_score表示的是每个query与其匹配到的target的IoU。这个匹配关系是由Hungarian Matcher决定的
    # 经过Hungarian Matcher匹配后：
    # query_5 匹配到了 target_0，IoU = 0.8
    # query_23 匹配到了 target_1，IoU = 0.7
    # 其他query都是负样本，IoU = 0
    # gt_score = [0, 0, 0, 0, 0, 0.8, 0, ..., 0, 0.7, 0, ...]
    # #                      ^query_5        ^query_23
    prob = inputs.sigmoid().detach()
    # calculate iou_aware_score and constrain the value following original implementation
    iou_aware_score = prob**k * gt_score.unsqueeze(-1)**(1 - k)
    iou_aware_score = iou_aware_score.clamp(min=0.01)
    target_score = targets * iou_aware_score
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + targets
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes
