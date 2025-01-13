from collections import defaultdict
import numpy as np
import torch
import logging
import os


class MatchingMonitor:
    def __init__(self, matcher):
        """
        初始化匹配监控器

        Args:
            matcher: DETR的matcher实例
        """
        self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        self.batch_count = 0
        self.matcher = matcher
        # 存储每个transition的统计信息
        self.accumulated_stats = defaultdict(lambda: {
            "total_queries": 0,          # 总query数
            "changed_queries": 0,        # 变化的query数
            "change_ratios": [],         # 每个batch的变化比例
            "per_class": defaultdict(lambda: {
                "total_queries": 0,
                "changed_queries": 0,
                "ratios": []
            })
        })

    def process_batch(self, outputs, targets):
        """
        处理一个batch的输出

        Args:
            outputs: 模型输出，包含最终输出和aux_outputs
            targets: 目标列表
        """
        self.batch_count += 1
        layer_matches = {}

        # 获取最终层匹配
        layer_matches['decoder_final'] = self._get_matches(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            targets
        )

        # 获取中间层匹配
        if 'aux_outputs' in outputs:
            for idx, aux_out in enumerate(outputs['aux_outputs']):
                layer_matches[f'decoder_{idx}'] = self._get_matches(
                    aux_out['pred_logits'],
                    aux_out['pred_boxes'],
                    targets
                )

        # 更新统计信息
        self._update_stats(layer_matches, targets)
        if self.batch_count % 1000 == 0:
            self.report_statistics()

    def _get_matches(self, pred_logits, pred_boxes, targets):
        """获取匹配关系"""
        return [self.matcher(b, l, gt['boxes'], gt['labels'])
                for b, l, gt in zip(pred_boxes, pred_logits, targets)]

    def _calculate_query_changes(self, curr_queries: set, next_queries: set):
        """
        计算实际的query匹配变化数量

        Args:
            curr_queries: 当前层匹配的query集合
            next_queries: 下一层匹配的query集合

        Returns:
            changed: 发生变化的数量
        """
        min_size = min(len(curr_queries), len(next_queries))
        common = len(curr_queries & next_queries)
        changed = min_size - common
        return changed, min_size

    def _update_stats(self, layer_matches, targets):
        """更新统计信息"""
        layers = sorted(layer_matches.keys())

        for i in range(len(layers) - 1):
            curr_layer = layers[i]
            next_layer = layers[i + 1]
            transition = f"{curr_layer}->{next_layer}"

            batch_total_queries = 0
            batch_changed_queries = 0
            class_stats = defaultdict(lambda: {"total_queries": 0, "changed_queries": 0})

            # 遍历batch中的每张图片
            for (curr_matches, next_matches, target) in zip(
                layer_matches[curr_layer],
                layer_matches[next_layer],
                targets
            ):
                curr_pred_idx, curr_tgt_idx = curr_matches
                next_pred_idx, next_tgt_idx = next_matches
                target_labels = target['labels']

                # 统计每个目标的匹配变化
                for tgt_idx in torch.unique(curr_tgt_idx):
                    class_id = target_labels[tgt_idx].item()

                    curr_queries = set(curr_pred_idx[curr_tgt_idx == tgt_idx].tolist())
                    next_queries = set(next_pred_idx[next_tgt_idx == tgt_idx].tolist())

                    # 计算变化数量
                    n_changed, n_total = self._calculate_query_changes(curr_queries, next_queries)

                    # 更新总体统计
                    batch_total_queries += n_total
                    batch_changed_queries += n_changed

                    # 更新类别统计
                    class_stats[class_id]["total_queries"] += n_total
                    class_stats[class_id]["changed_queries"] += n_changed

            # 计算当前batch的变化比例
            batch_ratio = (batch_changed_queries / batch_total_queries * 100) if batch_total_queries > 0 else 0

            # 更新累积统计
            stats = self.accumulated_stats[transition]
            stats["total_queries"] += batch_total_queries
            stats["changed_queries"] += batch_changed_queries
            stats["change_ratios"].append(batch_ratio)

            # 更新每个类别的统计
            for class_id, class_stat in class_stats.items():
                per_class = stats["per_class"][class_id]
                per_class["total_queries"] += class_stat["total_queries"]
                per_class["changed_queries"] += class_stat["changed_queries"]
                if class_stat["total_queries"] > 0:
                    class_ratio = (class_stat["changed_queries"] / class_stat["total_queries"] * 100)
                    per_class["ratios"].append(class_ratio)

    def report_statistics(self):
        """输出统计结果"""
        self.logger.info("\n=== Matching Change Statistics ===")

        for transition, stats in sorted(self.accumulated_stats.items()):
            self.logger.info(f"\n{transition}:")

            # 输出总体统计
            total = stats["total_queries"]
            changed = stats["changed_queries"]
            overall_ratio = (changed / total * 100) if total > 0 else 0
            mean_ratio = np.mean(stats["change_ratios"])
            std_ratio = np.std(stats["change_ratios"])

            self.logger.info(f"Overall statistics:")
            self.logger.info(f"  Total queries: {total}")
            self.logger.info(f"  Changed queries: {changed}")
            self.logger.info(f"  Overall change ratio: {overall_ratio:.2f}%")
            self.logger.info(f"  Average batch change ratio: {mean_ratio:.2f}% ± {std_ratio:.2f}%")

            # # 输出每个类别的统计
            # self.logger.info("\nPer-class statistics:")
            # for class_id, class_stats in sorted(stats["per_class"].items()):
            #     class_total = class_stats["total_queries"]
            #     class_changed = class_stats["changed_queries"]
            #     class_ratio = (class_changed / class_total * 100) if class_total > 0 else 0
            #     class_mean = np.mean(class_stats["ratios"]) if class_stats["ratios"] else 0
            #     class_std = np.std(class_stats["ratios"]) if class_stats["ratios"] else 0

            #     self.logger.info(f"  Class {class_id}:")
            #     self.logger.info(f"    Total queries: {class_total}")
            #     self.logger.info(f"    Changed queries: {class_changed}")
            #     self.logger.info(f"    Overall change ratio: {class_ratio:.2f}%")
            #     self.logger.info(f"    Average change ratio: {class_mean:.2f}% ± {class_std:.2f}%")

    def reset(self):
        """重置统计数据"""
        self.accumulated_stats.clear()