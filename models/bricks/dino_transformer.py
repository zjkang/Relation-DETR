import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.position_encoding import get_sine_pos_embed
from models.bricks.relation_transformer import (
    PositionRelationEmbedding,
    RelationTransformerDecoderLayer,
    RelationTransformerEncoderLayer,
)
from util.misc import inverse_sigmoid


class DINOTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        group_query_interaction: nn.Module = None, #zz
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder

        self.group_query_interaction = group_query_interaction #zz
        # self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)

        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        # nn.init.normal_(self.tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()

        tgt_embed, group_outputs_weights = self.group_query_interaction(memory) #zz
        target = tgt_embed.expand(multi_level_feats[0].shape[0], -1, -1) #zz
        # target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, group_outputs_weights #zz

    # zz
    def compute_spec_losses(self, group_weights):
        return self.group_query_interaction.compute_spec_losses(group_weights)

DINOTransformerEncoderLayer = RelationTransformerEncoderLayer


class DINOTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )

        return query


class DINOTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)
        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :], self.embed_dim // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # # NOTE: Here we integrate position_relation_embedding into DINO
            # src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
            # tgt_boxes = output_coord
            # pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
            # if attn_mask is not None:
            #     pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


DINOTransformerDecoderLayer = RelationTransformerDecoderLayer



class GroupQueryInteraction(nn.Module):
    def __init__(self, d_model, num_queries, num_groups=300):
        super().__init__()
        self.d_model = d_model
        self.num_specialized = num_groups
        self.num_queries = num_queries

        # 专门化query模板
        self.specialized_queries = nn.Embedding(num_groups, d_model)

        # 简化的特征处理器
        self.img_feature_pooling = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

        # 从图像特征生成权重的网络
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.num_queries * self.num_specialized)
        )

        nn.init.normal_(self.specialized_queries.weight)

    def forward(self, memory, pos=None):
        bs = memory.shape[0]
        # 1. 从图像特征生成权重矩阵
        image_feat = memory.mean(dim=1)  # [bs, d_model]
        image_feat = self.img_feature_pooling(image_feat)

        # 2. 生成组合权重
        weights = self.weight_generator(image_feat)
        weights = weights.view(bs, self.num_queries, self.num_specialized)
        weights = F.softmax(weights, dim=-1)

        # 使用权重组合组特征
        enahcned_queries = torch.matmul(weights, self.specialized_queries.weight)  # [bs, num_queries, d_model]

        return enahcned_queries, weights

    def compute_spec_losses(self, group_weights):
        # 计算specialized queries之间的相似度
        queries = F.normalize(self.specialized_queries.weight, dim=-1)  # 归一化
        # 计算余弦相似度
        similarity = F.cosine_similarity(
            queries.unsqueeze(1),  # [num_groups, 1, d_model]
            queries.unsqueeze(0),  # [1, num_groups, d_model]
            dim=-1
        )
        # 400.0: initial value 2.0 testing
        # 600.0: initial value 3.0
        # 1000.0: inital value 5.0
        scale_factor = 600.0
        # 移除对角线上的自相似度
        mask = torch.eye(self.num_specialized, device=queries.device)
        similarity = similarity * (1 - mask)
        diversity_loss = scale_factor * similarity.abs().sum() / (self.num_specialized * (self.num_specialized - 1))

        # only loss consider similarity > 0.5
        # threshold = 0.5
        # high_similarity = F.relu(similarity - threshold)
        # diversity_loss = scale_factor * high_similarity.sum() / (self.num_specialized * (self.num_specialized - 1))

        # 计算concentration loss：鼓励每个查询更加专注于特定的组
        # concentration_loss = -(group_weights.max(dim=-1)[0]).mean()
        # 方案1：Top-k稀疏性损失
        num_specialized = group_weights.shape[-1]
        k = int(num_specialized * 0.3)  # 期望的活跃组数
        top_k_weights, _ = torch.topk(group_weights, k, dim=-1)  # 获取前k个最大权重
        sparsity_loss = (1 - top_k_weights.sum(dim=-1)).mean()  # 鼓励top-k权重之和接近1

        loss_dict = {
            "loss_spec_diversity": diversity_loss,
            "loss_spec_l1": 10 * sparsity_loss
        }
        return loss_dict


# # 可选：添加diversity loss鼓励不同query的注意力模式不同
# def diversity_loss(self, attn_weights):
#     similarity = torch.matmul(attn_weights, attn_weights.transpose(-2, -1))
#     diversity_loss = torch.triu(similarity, diagonal=1).sum()
#     return diversity_loss

# # 可选：添加sparsity约束
# def sparsity_constraint(self, combination_weights):
#     return torch.norm(combination_weights, p=1)

# 正交约束
# 对比学习
