from collections import defaultdict
import numpy as np
import torch
import logging
import os


class MatchingMonitor:
    def __init__(self, matcher, matching_copies=None):
        """
        初始化匹配监控器

        Args:
            matcher: DETR的matcher实例
        """
        self.logger = logging.getLogger(os.path.basename(os.getcwd()) + "." + __name__)
        self.batch_count = 0
        self.matcher = matcher
        self.matching_copies = matching_copies if matching_copies is not None else [1,1,1,1,1,1]
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
        # 新增：每层每个类别使用的query统计
        self.class_query_stats = defaultdict(lambda: defaultdict(set))  # {layer: {class_id: set(queries)}}
        # 新增：每层每个query预测的类别统计
        self.query_class_stats = defaultdict(lambda: defaultdict(set))  # {layer: {query_id: set(classes)}}


    def process_batch(self, outputs, targets):
        """
        处理一个batch的输出

        Args:
            outputs: 模型输出，包含最终输出和aux_outputs
            targets: 目标列表
        """
        layer_matches = {}

        # 获取最终层匹配
        layer_matches['decoder_final'] = self._get_matches(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            targets,
            gt_copy=self.matching_copies[-1]
        )

        # 获取中间层匹配
        if 'aux_outputs' in outputs:
            for idx, aux_out in enumerate(outputs['aux_outputs']):
                layer_matches[f'decoder_{idx}'] = self._get_matches(
                    aux_out['pred_logits'],
                    aux_out['pred_boxes'],
                    targets,
                    gt_copy=self.matching_copies[idx]
                )

        # 更新统计信息
        self._update_stats(layer_matches, targets)
        self._update_query_usage_stats(layer_matches, targets)  # 累积query使用统计
        self.batch_count += 1
        if self.batch_count % 1000 == 0:
            self.report_statistics()
            self.report_query_usage_statistics()


    # assume many-to-one matching layer by layer in non-increasing order
    # matching_copies None: no extra copies
    # [2,2,2,2,2,2,1]: current logic
    def _get_matches(self, pred_logits, pred_boxes, targets, gt_copy=1):
        """获取匹配关系"""
        return [self.matcher(b, l, gt['boxes'], gt['labels'], gt_copy=gt_copy)
                for b, l, gt in zip(pred_boxes, pred_logits, targets)]


    def _calculate_query_changes(self, curr_queries: set, next_queries: set):
        """
        计算实际的query匹配变化数量
        if curr_queries = [1,3], next_queries = [1],可以认为没有发生匹配变化,因为至少有一个query匹配到了
        then no change, changed = 0, min_size = 1

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
        self.class_query_stats.clear()
        self.query_class_stats.clear()

    def _update_query_usage_stats(self, layer_matches, targets):
        """统计每层的query使用情况"""
        for layer_name, batch_matches in layer_matches.items():
            # 遍历batch中的每张图片
            for matches, target in zip(batch_matches, targets):
                pred_idx, tgt_idx = matches
                target_labels = target['labels']

                # 遍历每个目标
                for tgt_idx_i in torch.unique(tgt_idx):
                    class_id = target_labels[tgt_idx_i].item()
                    matched_queries = pred_idx[tgt_idx == tgt_idx_i].tolist()

                    # 更新类别使用的query统计
                    self.class_query_stats[layer_name][class_id].update(matched_queries)

                    # 更新query预测的类别统计
                    for query_id in matched_queries:
                        self.query_class_stats[layer_name][query_id].add(class_id)

    def report_query_usage_statistics(self):
        """报告query使用情况的统计"""
        self.logger.info("\n=== Query Usage Statistics ===")

        # 按层输出统计信息
        for layer_name in sorted(self.class_query_stats.keys()):
            self.logger.info(f"\nLayer: {layer_name}")

            # 1. 每个类别使用的query数量
            self.logger.info("\nQueries per class:")
            for class_id, queries in sorted(self.class_query_stats[layer_name].items()):
                self.logger.info(f"  Class {class_id}: {len(queries)} unique queries")
                self.logger.info(f"    Query IDs: {sorted(queries)}")

            # 2. 每个query预测的类别数量
            self.logger.info("\nClasses per query:")
            query_stats = defaultdict(int)  # 统计预测多个类别的query数量
            for query_id, classes in sorted(self.query_class_stats[layer_name].items()):
                n_classes = len(classes)
                query_stats[n_classes] += 1
                self.logger.info(f"  Query {query_id}: {n_classes} classes")
                self.logger.info(f"    Class IDs: {sorted(classes)}")

            # 输出query多样性统计
            self.logger.info("\nQuery diversity statistics:")
            for n_classes, count in sorted(query_stats.items()):
                self.logger.info(f"  {count} queries predicted {n_classes} different classes")

            # 计算一些汇总统计
            total_queries = len(self.query_class_stats[layer_name])
            total_classes = len(self.class_query_stats[layer_name])
            avg_classes_per_query = sum(len(classes) for classes in self.query_class_stats[layer_name].values()) / total_queries if total_queries > 0 else 0
            avg_queries_per_class = sum(len(queries) for queries in self.class_query_stats[layer_name].values()) / total_classes if total_classes > 0 else 0

            self.logger.info("\nSummary:")
            self.logger.info(f"  Total unique queries used: {total_queries}")
            self.logger.info(f"  Total classes: {total_classes}")
            self.logger.info(f"  Average classes per query: {avg_classes_per_query:.2f}")
            self.logger.info(f"  Average queries per class: {avg_queries_per_class:.2f}")